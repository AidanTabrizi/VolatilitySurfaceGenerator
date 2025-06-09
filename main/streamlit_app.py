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
    /* Hide the Streamlit header */
    header {visibility: hidden;}

    /* Hide the Streamlit footer */
    footer {visibility: hidden;}

    /* Hide the hamburger menu */
    .css-1y4p8pa {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ───────────────────────────  Helper functions  ────────────────────────
def fetch_stock_data(ticker: str, ref_date: datetime.date, max_tries=5):
    """Download most-recent price data, stepping back if holiday."""
    for _ in range(max_tries):
        df = yf.download(ticker, start=ref_date)
        if not df.empty:
            return df
        ref_date -= timedelta(days=1)
    return None

def recent_market_day() -> datetime.date:
    today = datetime.today().date()
    if today.weekday() == 5:  # Saturday
        return today - timedelta(days=1)
    if today.weekday() == 6:  # Sunday
        return today - timedelta(days=2)
    return today

# ─────────────────────────  Core IV-surface solver  ────────────────────
def volatility_solver(ticker: str, rfr: float, opt_type: str,
                      sigma0: float, tol: float):
    # 1) Spot price
    trade_date = recent_market_day()
    spot_df    = fetch_stock_data(ticker, trade_date)
    if spot_df is None or spot_df.empty:
        st.error("Couldn’t fetch recent price data.")
        return None, None
    S0   = float(spot_df["Close"].iloc[-1])
    today = spot_df.index[-1]

    # 2) Option chain
    tkr = yf.Ticker(ticker)
    rows = []
    for exp in tkr.options:
        chain = tkr.option_chain(exp)
        for (_, row) in chain.calls.iterrows():
            rows.append([exp, row["strike"], (row["bid"]+row["ask"])/2, "CALL"])
        for (_, row) in chain.puts.iterrows():
            rows.append([exp, row["strike"], (row["bid"]+row["ask"])/2, "PUT"])

    if not rows:
        st.error("No option data returned.")
        return None, None

    df = pd.DataFrame(rows,
                      columns=["expiry", "strike", "midprice", "type"])
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df.dropna(subset=["strike", "midprice"], inplace=True)
    df.drop_duplicates(subset=["expiry", "strike", "type"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3) keep strikes ±20 % of spot  (NumPy mask ⇒ never re-indexes)
    strikes = df["strike"].to_numpy(dtype=float)
    keep_strike = np.logical_and(strikes > 0.8*S0, strikes < 1.2*S0)
    df = df.iloc[keep_strike]

    # 4) convert expiry to days-to-expiry and keep 1-99 days
    df["dte"] = (pd.to_datetime(df["expiry"]) - today).dt.days
    dte_arr = df["dte"].to_numpy(dtype=int)
    keep_dte = np.logical_and(dte_arr > 0, dte_arr < 100)
    df = df.iloc[keep_dte]

    if df.empty:
        st.error("No options within strike/DTE filters.")
        return None, None

    # 5) pivot to (dte, strike) index  ×  CALL / PUT columns
    df = (df
          .pivot_table(index=["dte", "strike"], columns="type",
                       values="midprice")
          .sort_index())

    # ───── Black-Scholes + Greeks ──────────────────────────────────────
    def bs_residual(sig, S, K, P, T, r, otype):
        d1 = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        if otype == "CALL":
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) - P
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - P - S*norm.cdf(-d1)

    iv_list, greek_list = [], []
    for (dte, K), price in df[opt_type].items():
        T = dte / 365.0
        try:
            iv = float(fsolve(bs_residual, sigma0,
                              args=(S0, K, price, T, rfr, opt_type),
                              xtol=tol)[0])
        except Exception:
            iv = np.nan
        # Greeks (only if iv solved)
        if np.isnan(iv) or iv <= 0:
            iv_list.append(np.nan)
            greek_list.append([np.nan]*5)
            continue
        d1 = (np.log(S0/K) + (rfr + 0.5*iv**2)*T) / (iv*np.sqrt(T))
        d2 = d1 - iv*np.sqrt(T)
        if opt_type == "CALL":
            delta = norm.cdf(d1)
            rho   = K*T*np.exp(-rfr*T)*norm.cdf(d2)/100
        else:
            delta = norm.cdf(d1) - 1
            rho   = -K*T*np.exp(-rfr*T)*norm.cdf(-d2)/100
        gamma = norm.pdf(d1)/(S0*iv*np.sqrt(T))
        theta = -(S0*norm.pdf(d1)*iv/(2*np.sqrt(T))
                  - rfr*K*np.exp(-rfr*T)*(norm.cdf(d2) if opt_type=="CALL"
                                          else norm.cdf(-d2)))/365
        vega  = S0*norm.pdf(d1)*np.sqrt(T)/100
        iv_list.append(iv)
        greek_list.append([delta, gamma, theta, vega, rho])

    iv_series = pd.Series(iv_list, index=df[opt_type].index)
    greek_df  = pd.DataFrame(greek_list,
                             index=df[opt_type].index,
                             columns=["DELTA","GAMMA","THETA","VEGA","RHO"])

    iv_grid  = iv_series.unstack(level=0).interpolate("linear")
    greek_grid = greek_df.unstack(level=0).interpolate("linear").fillna(0)
    return iv_grid, greek_grid

# ─────────────────────────────  Surface plot  ──────────────────────────
def plot_surface(vol_surf: pd.DataFrame, greek_surf: pd.DataFrame,
                 greek: str, ticker: str, opt_type: str):
    X = vol_surf.columns.values      # days-to-exp
    Y = vol_surf.index.values        # strikes
    X, Y = np.meshgrid(X, Y)
    Z = vol_surf.values
    C = greek_surf[greek].values

    if greek == "DELTA":
        cmap, norm = ("plasma_r" if opt_type=="PUT" else "plasma",
                      Normalize(-1 if opt_type=="PUT" else 0,
                                0 if opt_type=="PUT" else 1))
    elif greek == "GAMMA":
        cmap, norm = "plasma", Normalize(0, np.nanmax(C))
    elif greek == "THETA":
        cmap, norm = "plasma_r", Normalize(np.nanmin(C), 0)
    elif greek == "VEGA":
        cmap, norm = "plasma", Normalize(0, np.nanmax(C))
    else:  # RHO
        cmap, norm = ("plasma_r" if opt_type=="PUT" else "plasma",
                      Normalize(np.nanmin(C), np.nanmax(C)))

    custom_style = {
        'axes.facecolor': '#0E1118',  # Background color of the plot
        'axes.edgecolor': '#FFFFFF',  # Edge color of the plot
        'axes.labelcolor': '#FFFFFF',  # Color of x, y, z axis labels
        'figure.facecolor': '#0E1118',  # Background color of the figure
        'grid.color': '#555555',  # Color of grid lines, slightly brighter for better contrast
        'text.color': '#FFFFFF',  # Text color
        'axes.titleweight': 'bold',  # Title weight
        'axes.labelweight': 'bold',  # Label weight
        'axes.titlesize': 20,  # Title size
        'axes.labelsize': 12,  # Label size
        'font.family': 'sans-serif',  # Font family
        'font.size': 11,  # Font size
        'legend.fontsize': 11,  # Legend font size
        'figure.autolayout': True,  # Automatically adjust the layout
    }
                     
    plt.rcParams.update(custom_style)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#555555')
    ax.yaxis.pane.set_edgecolor('#555555')
    ax.zaxis.pane.set_edgecolor('#555555')
    ax.tick_params(axis='x', colors='#FFFFFF')
    ax.tick_params(axis='y', colors='#FFFFFF')
    ax.tick_params(axis='z', colors='#FFFFFF')
    ax.xaxis.set_tick_params(labelcolor='#FFFFFF')
    ax.yaxis.set_tick_params(labelcolor='#FFFFFF')
    ax.zaxis.set_tick_params(labelcolor='#FFFFFF')
                     

    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.get_cmap(cmap)(norm(C)), rstride=1,cstride=1, edgecolor='#657383', linewidth=0.02, antialiased=False)
    ax.set_xlabel("Days to Expiry")
    ax.set_ylabel("Strike")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"{ticker.upper()} {opt_type} Implied Volatility Surface\n(coloured by {greek})")
    m  = plt.cm.ScalarMappable(cmap=cmap, norm=norm);  m.set_array(C)
    color_bar = fig.colorbar(m, shrink=0.5, aspect=8, label=greek)
    color_bar.set_label(f'{greek_parameter.capitalize()}', color='#FFFFFF', fontsize=12, labelpad=15, weight='bold')

    # Set the tick parameters (optional customization)
    color_bar.ax.tick_params(labelsize=10, labelcolor='#FFFFFF')
    st.pyplot(fig)

# ─────────────────────────────  Streamlit UI  ──────────────────────────
# Streamlit UI
with st.sidebar:
    st.title("Volatility Surface Generator")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/aidan-tabrizi/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Aidan Tabrizi`</a>', unsafe_allow_html=True)

# Input fields in the sidebar
ticker = st.sidebar.text_input('Ticker Symbol:', value='AAPL')
option_type = st.sidebar.selectbox('Option Type:', ['CALL', 'PUT'])
greek_parameter = st.sidebar.selectbox('Heatmap Parameter:', ['DELTA','GAMMA','THETA','VEGA','RHO'])
risk_free_rate = st.sidebar.number_input("Risk-Free Rate:", value=0.04)
# Added input field for initial volatility guess
sigma_guess = st.sidebar.number_input("Initial Volatility Guess:", value=0.4, step=0.01)
tolerance = 1e-9  # Tolerance for the solver
st.info("Visualize the volatility surface and option Greeks (Delta, Gamma, Theta, Vega, Rho) for a call or put option of any chosen security! Just enter the ticker symbol, select the option type, input the risk-free rate, provide an initial guess for volatility, and choose a Greek parameter to overlay on the surface. Using market option prices from Yahoo Finance, the implied volatility is calculated with the Black-Scholes model, and the results are plotted via Matplotlib with interactive heatmaps to enhance analysis and understanding.")

# Main Content

if ticker:
    with st.spinner("Computing surface…"):
        iv_grid, gk_grid = volatility_solver(ticker, risk_free_rate, option_type, sigma_guess, tolerance)
    if iv_grid is not None and not iv_grid.empty:
        st.success("Done!")
        plot_surface(iv_grid, gk_grid, greek_parameter, ticker, option_type)
