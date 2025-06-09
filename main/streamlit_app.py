import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ──────────────────────────────────  STREAMLIT CHROME  ─────────────────────────
st.markdown(
    """
    <style>
        header, footer, .css-1y4p8pa {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────  HELPERS  ───────────────────────────────────
def fetch_stock_data(ticker, start_date, max_attempts=5):
    """Download most-recent market data, stepping back a day if holiday."""
    for _ in range(max_attempts):
        data = yf.download(ticker, start=start_date)
        if not data.empty:
            return data
        start_date -= timedelta(days=1)
    return None

def get_recent_market_day() -> datetime.date:
    today = datetime.today().date()
    if today.weekday() == 5:        # Saturday
        return today - timedelta(days=1)
    if today.weekday() == 6:        # Sunday
        return today - timedelta(days=2)
    return today

# ───────────────────────────────  CORE SOLVER  ────────────────────────────────
def volatility_solver(ticker, rfr, option_type, sigma0, tol):
    # 1) Spot price
    trade_date = get_recent_market_day()
    spot_df    = fetch_stock_data(ticker, trade_date)
    if spot_df is None or spot_df.empty:
        st.error("No recent price data found.")
        return None, None
    S0   = spot_df["Close"].iloc[-1]
    today = spot_df.index[-1]

    # 2) Pull option chains for every expiry
    rows = []
    tk   = yf.Ticker(ticker)
    for expiry in tk.options:
        oc = tk.option_chain(expiry)
        for _, opt in oc.calls.iterrows():
            rows.append([expiry, opt["strike"], (opt["bid"] + opt["ask"]) / 2, "CALL"])
        for _, opt in oc.puts.iterrows():
            rows.append([expiry, opt["strike"], (opt["bid"] + opt["ask"]) / 2, "PUT"])

    if not rows:
        st.error("No option data returned by yfinance.")
        return None, None

    # ----------------  CLEAN & FILTER (index-safe)  -----------------
    df = pd.DataFrame(rows, columns=["expiry", "strike", "price", "type"])
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df.dropna(subset=["strike", "price"], inplace=True)
    df.reset_index(drop=True, inplace=True)                 # RangeIndex

    # keep strikes ±20 %
    mask_strike = (df["strike"].to_numpy() > S0*0.8) & (df["strike"].to_numpy() < S0*1.2)
    df = df.iloc[mask_strike]

    # days-to-expiry
    df["dte"] = (pd.to_datetime(df["expiry"]) - today).dt.days
    # keep 1–99 days
    mask_exp = (df["dte"].to_numpy() > 0) & (df["dte"].to_numpy() < 100)
    df = df.iloc[mask_exp]

    if df.empty:
        st.error("No options left after filtering by strike/expiry.")
        return None, None

    # Pivot to (dte, strike) index with CALL / PUT columns
    df = df.pivot_table(index=["dte", "strike"], columns="type", values="price").sort_index()

    # ----------------  Black-Scholes helpers  -----------------
    def bs_price(vol, S, K, Pmkt, T, r, cp):
        d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        if cp == "CALL":
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) - Pmkt
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - Pmkt - S*norm.cdf(-d1)

    def greeks(S, K, T, r, vol, cp):
        d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        if cp == "CALL":
            delta = norm.cdf(d1); rho = K*T*np.exp(-r*T)*norm.cdf(d2)/100
        else:
            delta = norm.cdf(d1) - 1; rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100
        gamma = norm.pdf(d1)/(S*vol*np.sqrt(T))
        theta = -(S*norm.pdf(d1)*vol/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*(
                 norm.cdf(d2) if cp=="CALL" else norm.cdf(-d2)))/365
        vega  = S*norm.pdf(d1)*np.sqrt(T)/100
        return delta, gamma, theta, vega, rho

    # ----------------  Solve for IV + Greeks  ----------------
    iv_list, greek_list = [], []
    for (dte, K), price in df[option_type].dropna().items():
        T = dte / 365.0
        try:
            iv = fsolve(bs_price, x0=sigma0,
                        args=(S0, K, price, T, rfr, option_type), xtol=tol)[0]
            if 0 < iv < 5:   # sanity
                iv_list.append(iv)
                greek_list.append(greeks(S0, K, T, rfr, iv, option_type))
            else:
                iv_list.append(np.nan); greek_list.append([np.nan]*5)
        except Exception:
            iv_list.append(np.nan); greek_list.append([np.nan]*5)

    iv_series  = pd.Series(iv_list, index=df[option_type].dropna().index)
    greeks_df  = pd.DataFrame(greek_list, index=iv_series.index,
                              columns=["DELTA","GAMMA","THETA","VEGA","RHO"])

    iv_grid     = iv_series.unstack(0).interpolate("linear")
    greeks_grid = greeks_df.unstack(0).interpolate("linear").fillna(0)

    return iv_grid, greeks_grid

# ─────────────────────────────────  PLOTTER  ───────────────────────────────────
def plot_surface(vol_grid, greek_grid, greek_param, option_type, ticker):
    X = vol_grid.columns.values          # days-to-expiry
    Y = vol_grid.index.values            # strikes
    X, Y = np.meshgrid(X, Y)
    Z = vol_grid.values
    C = greek_grid[greek_param].values

    # simple colour-scale
    if greek_param == "DELTA":
        norm = Normalize(-1, 0) if option_type=="PUT" else Normalize(0, 1)
    elif greek_param == "GAMMA":
        norm = Normalize(0, np.nanmax(C))
    elif greek_param == "THETA":
        norm = Normalize(np.nanmin(C), 0)
    elif greek_param == "VEGA":
        norm = Normalize(0, np.nanmax(C))
    else:  # RHO
        norm = Normalize(-0.5,0) if option_type=="PUT" else Normalize(0,0.5)

    fig = plt.figure(figsize=(14,7))
    ax  = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("plasma_r" if (greek_param in ["DELTA","THETA","RHO"]
                                       and option_type=="PUT") else "plasma")
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(C)), linewidth=0, antialiased=False)
    ax.set_xlabel("Days to Expiry"); ax.set_ylabel("Strike"); ax.set_zlabel("IV")
    ax.set_title(f"{ticker.upper()} {option_type} – IV Surface w/ {greek_param}")
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm); m.set_array(C)
    fig.colorbar(m, shrink=0.5).set_label(greek_param)
    st.pyplot(fig)

# ────────────────────────────────  SIDEBAR  ────────────────────────────────
with st.sidebar:
    st.title("Volatility Surface Generator")
    st.write("Created by [Aidan Tabrizi](https://www.linkedin.com/in/aidan-tabrizi/)")
    ticker        = st.text_input("Ticker", value="AAPL")
    option_type   = st.selectbox("Option Type", ["CALL","PUT"])
    greek_param   = st.selectbox("Greek Heat-map", ["DELTA","GAMMA","THETA","VEGA","RHO"])
    risk_free     = st.number_input("Risk-free Rate", value=0.04)
    sigma_guess   = st.number_input("Initial Vol Guess", value=0.4, step=0.01)

# ────────────────────────────────  MAIN  ───────────────────────────────────
if ticker:
    with st.spinner("Crunching…"):
        iv_grid, gk_grid = volatility_solver(ticker, risk_free,
                                             option_type, sigma_guess, 1e-8)
    if iv_grid is not None and not iv_grid.empty:
        st.success("Done!")
        plot_surface(iv_grid, gk_grid, greek_param, option_type, ticker)
    else:
        st.error("Could not build IV surface – check data filters or ticker.")
else:
    st.info("Enter a ticker symbol to begin.")
