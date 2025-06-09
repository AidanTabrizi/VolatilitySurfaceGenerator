# ──────────────────────────────  Imports  ──────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ───────────────────────────  Streamlit chrome  ────────────────────────
st.markdown(
    """
    <style>
    /* Hide the Streamlit header, footer, and hamburger menu */
    header, footer, .css-1y4p8pa {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────  Helper functions  ────────────────────────
def fetch_stock_data(ticker: str, ref_date: datetime.date, max_tries: int = 5):
    """Download most-recent price data, stepping back a day if a holiday/weekend."""
    for _ in range(max_tries):
        # yfinance downloads data up to the day before the start date, so we request from ref_date
        df = yf.download(ticker, start=ref_date, progress=False)
        if not df.empty:
            return df
        ref_date -= timedelta(days=1)
    return None


def recent_market_day() -> datetime.date:
    """Finds the most recent day the market was open."""
    today = datetime.today().date()
    if today.weekday() == 5:  # Saturday → Friday
        return today - timedelta(days=1)
    if today.weekday() == 6:  # Sunday   → Friday
        return today - timedelta(days=2)
    return today


# ─────────────────────────  Core IV-surface solver  ────────────────────
def volatility_solver(ticker: str, rfr: float, opt_type: str, sigma0: float, tol: float):
    # 1) Spot price -----------------------------------------------------
    trade_date = recent_market_day()
    spot_df = fetch_stock_data(ticker, trade_date)
    if spot_df is None or spot_df.empty:
        st.error(f"Couldn’t fetch recent price data for {ticker.upper()}.")
        return None, None
    S0 = float(spot_df["Close"].iloc[-1])
    today = spot_df.index[-1]

    # 2) Option chain ---------------------------------------------------
    rows = []
    tkr = yf.Ticker(ticker)
    try:
        expirations = tkr.options
        if not expirations:
            st.error(f"No option expiration dates found for {ticker.upper()}.")
            return None, None
    except Exception:
        st.error(f"Could not fetch option data for {ticker.upper()}. It may be an invalid ticker.")
        return None, None

    for exp in expirations:
        chain = tkr.option_chain(exp)
        # Combine calls and puts for efficient processing
        combined_df = pd.concat([
            chain.calls.assign(type="CALL"),
            chain.puts.assign(type="PUT")
        ])
        for _, row in combined_df.iterrows():
            rows.append([exp, row["strike"], (row["bid"] + row["ask"]) / 2, row["type"]])

    if not rows:
        st.error(f"No option contracts returned from Yahoo for {ticker.upper()}.")
        return None, None

    df = pd.DataFrame(rows, columns=["expiry", "strike", "midprice", "type"])
    df.dropna(subset=["strike", "midprice"], inplace=True)
    df.drop_duplicates(subset=["expiry", "strike", "type"], inplace=True)

    # 3) Keep strikes ±20 % of spot
    df = df[(df["strike"] > 0.8 * S0) & (df["strike"] < 1.2 * S0)]

    # 4) Convert expiry → days-to-expiry and keep 1–99 days
    df["dte"] = (pd.to_datetime(df["expiry"]) - today).dt.days
    df = df[(df["dte"] > 0) & (df["dte"] < 100)]

    if df.empty:
        st.error("No options available after applying strike and DTE filters.")
        return None, None

    # 5) Pivot to (dte, strike) index × CALL/PUT columns
    df = df.pivot_table(index=["dte", "strike"], columns="type", values="midprice").sort_index()
    if opt_type not in df.columns or df[opt_type].isnull().all():
        st.error(f"No valid {opt_type.lower()} prices found after filtering.")
        return None, None

    # 6) Black-Scholes + Greeks
    def bs_residual(sig, S, K, P, T, r, otype):
        if sig <= 0: return 1e9 # Ensure sigma is positive
        d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        if otype == "CALL":
            price_model = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price_model = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price_model - P

    iv_list, greek_list = [], []
    # Iterate over valid prices for the selected option type
    valid_options = df[opt_type].dropna()
    for (dte, K), price in valid_options.items():
        T = dte / 365.0
        try:
            iv = float(fsolve(bs_residual, sigma0, args=(S0, K, price, T, rfr, opt_type), xtol=tol)[0])
            if not (0 < iv < 5): iv = np.nan # Filter out unreasonable IV values
        except Exception:
            iv = np.nan

        if np.isnan(iv):
            iv_list.append(np.nan)
            greek_list.append([np.nan] * 5)
            continue

        d1 = (np.log(S0 / K) + (rfr + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)

        if opt_type == "CALL":
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-rfr * T) * norm.cdf(d2) / 100
            theta_sign = -1
        else: # PUT
            delta = norm.cdf(d1) - 1
            rho = -K * T * np.exp(-rfr * T) * norm.cdf(-d2) / 100
            theta_sign = 1
        
        gamma = norm.pdf(d1) / (S0 * iv * np.sqrt(T))
        theta = (-(S0 * norm.pdf(d1) * iv / (2 * np.sqrt(T))) + theta_sign * rfr * K * np.exp(-rfr * T) * norm.cdf(theta_sign * d2)) / 365
        vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100

        iv_list.append(iv)
        greek_list.append([delta, gamma, theta, vega, rho])

    iv_series = pd.Series(iv_list, index=valid_options.index).reindex(df.index)
    greek_df = pd.DataFrame(greek_list, index=valid_options.index, columns=["DELTA", "GAMMA", "THETA", "VEGA", "RHO"]).reindex(df.index)

    iv_grid = iv_series.unstack(level=0).interpolate(method="linear", axis=1)
    greek_grid = greek_df.unstack(level=0).interpolate(method="linear", axis=1).fillna(0)
    
    if iv_grid.empty or iv_grid.isnull().all().all():
        st.error("Could not construct volatility grid. IV calculation failed for all points.")
        return None, None
        
    return iv_grid, greek_grid


# ─────────────────────────────  Surface plot  ──────────────────────────
def plot_surface(vol_grid: pd.DataFrame, greek_grid: pd.DataFrame, greek: str, ticker: str, opt_type: str):
    # Custom styling for a professional look
    plt.style.use("dark_background")
    plt.rcParams.update({
        'axes.labelcolor': '#FFFFFF', 'axes.titlecolor': '#FFFFFF',
        'xtick.color': '#FFFFFF', 'ytick.color': '#FFFFFF',
        'axes.edgecolor': '#555555', 'grid.color': '#555555',
        'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
        'axes.titlesize': 20, 'axes.labelsize': 12, 'font.size': 11,
    })

    # Prepare data for meshgrid
    X = vol_grid.columns.values
    Y = vol_grid.index.values
    X, Y = np.meshgrid(X, Y)
    Z = vol_grid.values
    C = greek_grid[greek].values

    # Context-aware color mapping for Greeks
    if greek == "DELTA":
        cmap = "plasma_r" if opt_type == "PUT" else "plasma"
        norm = Normalize(-1 if opt_type == "PUT" else 0, 0 if opt_type == "PUT" else 1)
    elif greek == "GAMMA":
        cmap = "plasma"; norm = Normalize(0, np.nanmax(C) if np.nanmax(C) > 0 else 0.1)
    elif greek == "THETA":
        cmap = "plasma_r"; norm = Normalize(np.nanmin(C) if np.nanmin(C) < 0 else -0.1, 0)
    elif greek == "VEGA":
        cmap = "plasma"; norm = Normalize(0, np.nanmax(C) if np.nanmax(C) > 0 else 0.1)
    else:  # RHO
        cmap = "plasma_r" if opt_type == "PUT" else "plasma"
        norm = Normalize(np.nanmin(C), np.nanmax(C))

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Remove pane fills for a cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Plot the surface with colored faces and a subtle wireframe
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.get_cmap(cmap)(norm(C)), 
                           edgecolor='#657383', linewidth=0.05, antialiased=False)

    ax.set_xlabel('Time to Expiry (Days)')
    ax.set_ylabel('Strike Price (USD)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Volatility Surface with {greek.capitalize()} for {ticker.upper()} {opt_type.capitalize()} Options')
    
    # Add a color bar to explain the Greek heatmap
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(C)
    color_bar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    color_bar.set_label(f'{greek.capitalize()}', weight='bold')
    
    st.pyplot(fig)


# ─────────────────────────────  Streamlit UI  ──────────────────────────
with st.sidebar:
    st.title("Volatility Surface Generator")
    st.write("Created by:")
    linkedin_url = "https://www.linkedin.com/in/aidan-tabrizi/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">Aidan Tabrizi</a>', unsafe_allow_html=True)
    st.write("---")

    ticker = st.text_input('Ticker Symbol:', value='AAPL')
    opt_type = st.selectbox('Option Type:', ['CALL', 'PUT'])
    greek = st.selectbox('Heatmap Parameter:', ['DELTA', 'GAMMA', 'THETA', 'VEGA', 'RHO'])
    rfr = st.number_input("Risk-Free Rate:", value=0.04, format="%.3f", step=0.001)
    sigma_guess = st.number_input("Initial Volatility Guess:", value=0.4, step=0.01, format="%.2f")
    tol = 1e-9
    st.write("---")
    st.info("Visualize the volatility surface and option Greeks for any chosen security. Enter a ticker, select the option type, and choose a Greek to overlay on the surface as a heatmap. The app uses live market data to calculate implied volatility via the Black-Scholes model.")

# --- Main Content ---
if ticker:
    with st.spinner('Calculating implied volatility surface...'):
        iv_grid, gk_grid = volatility_solver(ticker, rfr, opt_type, sigma_guess, tol)

    if iv_grid is not None and not iv_grid.empty:
        st.success('Calculation complete!')
        plot_surface(iv_grid, gk_grid, greek, ticker, opt_type)
    else:
        # Errors are now shown inside the solver, but this catches final failures.
        st.warning("Could not generate a plot. Please check the ticker symbol and data availability.")
