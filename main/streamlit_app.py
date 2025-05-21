# ────────────────────────────  Imports  ────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ─────────────────────────  Streamlit Tweaks  ──────────────────────
st.markdown(
    """
    <style>
        header, footer, .css-1y4p8pa {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────  Utility: most-recent market day  ────────────
def most_recent_mkt_day() -> datetime.date:
    today = datetime.today().date()
    if today.weekday() == 5:     # Sat
        return today - timedelta(days=1)
    if today.weekday() == 6:     # Sun
        return today - timedelta(days=2)
    return today                 # Mon-Fri

# ────────────────────  Utility: fetch equity prices  ───────────────
def fetch_stock_data(tkr: str, start_day: datetime.date, max_tries: int = 5):
    for _ in range(max_tries):
        df = yf.download(tkr, start=start_day)
        if not df.empty:
            return df
        start_day -= timedelta(days=1)
    return None

# ───────────────────────  Volatility Solver  ───────────────────────
def volatility_solver(ticker, rfr, side, sigma_guess, tol):
    """Return (vol_surface, greeks_surface)  – both DataFrames"""
    # ----- underlying -----
    recent_day  = most_recent_mkt_day()
    stock_df    = fetch_stock_data(ticker, recent_day)
    if stock_df is None or stock_df.empty:
        st.error(f"No price data for {ticker}.")
        return None, None

    S0          = stock_df["Close"].iloc[-1]
    today       = stock_df.index[-1]

    # ----- option chain -----
    yftkr       = yf.Ticker(ticker)
    maturities  = yftkr.options
    rows        = []                                           # build list → DataFrame

    for exp in maturities:
        chain = yftkr.option_chain(exp)
        def mid(row): return (row["bid"] + row["ask"]) / 2

        for _, row in chain.calls.iterrows():
            rows.append([exp, row["strike"], mid(row), "CALL"])
        for _, row in chain.puts.iterrows():
            rows.append([exp, row["strike"], mid(row), "PUT"])

    df = (
        pd.DataFrame(
            rows, columns=["expiration", "strike", "midprice", "type"]
        )
        .dropna(subset=["midprice"])                                     # remove missing quotes
    )

    # ----- strike filter BEFORE pivot -----
    df = df[(df["strike"] > 0.8*S0) & (df["strike"] < 1.2*S0)]

    # ----- days to expiry & expiry filter BEFORE pivot -----
    df["days"] = pd.to_datetime(df["expiration"])
    df["days"] = (df["days"] - today).dt.days
    df = df[(df["days"] > 0) & (df["days"] < 100)]

    if df.empty:
        st.error("No options left after filtering.")
        return None, None

    # ----- pivot (index: days,strike | columns: CALL/PUT) -----
    df = (
        df.set_index(["days", "strike", "type"])
          .sort_index()
          .pivot_table(index=["days", "strike"], columns="type", values="midprice")
    )

    if side not in df.columns:
        st.error(f"No {side} data inside ±20 % strike window.")
        return None, None

    # ───────────  Black-Scholes helpers  ───────────
    def bs_resid(sigma, s0, k, p, t, r, _side):
        d1 = (np.log(s0/k) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
        d2 = d1 - sigma*np.sqrt(t)
        if _side == "CALL":
            model = s0*norm.cdf(d1) - k*np.exp(-r*t)*norm.cdf(d2)
        else:
            model = k*np.exp(-r*t)*norm.cdf(-d2) - s0*norm.cdf(-d1)
        return model - p

    def greeks(s0, k, t, r, sig, _side):
        d1 = (np.log(s0/k) + (r + 0.5*sig**2)*t) / (sig*np.sqrt(t))
        d2 = d1 - sig*np.sqrt(t)
        pdf = norm.pdf(d1)
        if _side == "CALL":
            delta = norm.cdf(d1)
            rho   =  k*t*np.exp(-r*t)*norm.cdf(d2)/100
        else:
            delta = norm.cdf(d1) - 1
            rho   = -k*t*np.exp(-r*t)*norm.cdf(-d2)/100
        gamma = pdf / (s0*sig*np.sqrt(t))
        vega  = s0*pdf*np.sqrt(t)/100
        theta = -(s0*pdf*sig)/(2*np.sqrt(t)) - r*k*np.exp(-r*t)*(
                 norm.cdf(d2) if _side=="CALL" else norm.cdf(-d2))
        theta /= 365
        return delta, gamma, theta, vega, rho

    # ───────────  solve vols + greeks  ───────────
    vols, g_rows = [], []
    for (t_days, k), price in df[side].items():
        T = t_days / 365
        try:
            vol = float(
                fsolve(bs_resid, sigma_guess,
                       args=(S0, k, price, T, rfr, side),
                       xtol=tol)[0]
            )
            vol = np.nan if not (0 < vol < 5) else vol
        except Exception:
            vol = np.nan
        vols.append(vol)
        g_rows.append(
            greeks(S0, k, T, rfr, vol, side) if not np.isnan(vol)
            else (np.nan,)*5
        )

    vol_ser  = pd.Series(vols, index=df[side].index)
    greeks_df= pd.DataFrame(
        g_rows, index=df[side].index,
        columns=["DELTA","GAMMA","THETA","VEGA","RHO"],
    )

    vol_surf   = vol_ser.unstack(0).interpolate("linear")
    greeks_surf= greeks_df.unstack(0).interpolate("linear").fillna(0)
    return vol_surf, greeks_surf

# ────────────────────  3-D Surface Plotter  ────────────────────────
def plot_surface(vol, greeks, greek, side, ticker):
    plt.rcParams.update({
        "axes.facecolor":   "#0E1118",
        "figure.facecolor": "#0E1118",
        "text.color":       "#FFFFFF",
        "axes.labelcolor":  "#FFFFFF",
        "axes.edgecolor":   "#FFFFFF",
        "grid.color":       "#555555",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize":   20,
        "font.size":        11,
        "legend.fontsize":  11,
    })

    X = vol.columns.values            # expiry (days)
    Y = vol.index.values              # strike
    X, Y = np.meshgrid(X, Y)
    Z = vol.values
    C = greeks[greek].values

    # colour-map normalisation
    if greek == "DELTA":
        vmin, vmax = (-1, 0) if side == "PUT" else (0, 1)
    elif greek == "GAMMA":
        vmin, vmax = C.min(), max(C.max(), 0.2)
    elif greek == "THETA":
        vmin, vmax = C.min(), 0
    elif greek == "VEGA":
        vmin, vmax = 0, max(C.max(), 0.5)
    else:  # RHO
        vmin, vmax = (-0.5, 0) if side == "PUT" else (0, 0.5)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "plasma_r" if (greek in {"DELTA","THETA","RHO"} and side=="PUT") else "plasma"

    fig = plt.figure(figsize=(16, 8))
    ax  = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z, facecolors=plt.cm.get_cmap(cmap)(norm(C)),
        rstride=1, cstride=1, edgecolor="#657383", linewidth=0.05, antialiased=False
    )

    ax.set_xlabel("Time to Expiry (Days)", weight="bold")
    ax.set_ylabel("Strike Price",         weight="bold")
    ax.set_zlabel("Implied Volatility",   weight="bold")
    ax.set_title(f"{ticker.upper()} {side} – IV Surface coloured by {greek}", weight="bold")

    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(C)
    cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(greek, color="#FFFFFF", weight="bold")

    st.pyplot(fig)

# ─────────────────────────  Sidebar (UI)  ──────────────────────────
with st.sidebar:
    st.title("Volatility Surface Generator")
    st.write("Created by:")
    st.markdown(
        '<a href="https://www.linkedin.com/in/aidan-tabrizi/" target="_blank">'
        '<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" '
        'width="25" height="25" style="vertical-align: middle;"> Aidan&nbsp;Tabrizi'
        '</a>', unsafe_allow_html=True
    )
    ticker      = st.text_input("Ticker Symbol:", value="AAPL")
    side        = st.selectbox("Option Type:", ["CALL", "PUT"])
    greek       = st.selectbox("Heatmap Parameter:", ["DELTA","GAMMA","THETA","VEGA","RHO"])
    rfr         = st.number_input("Risk-Free Rate:", value=0.04, step=0.001)
    sigma_guess = st.number_input("Initial Vol Guess:", value=0.40, step=0.01)
    tol         = 1e-9

    st.write("""
        Visualise the implied-volatility surface and overlay any Greek parameter.
        Data are fetched live from Yahoo Finance and solved with Black–Scholes.
    """)

# ───────────────────────────  Main Logic  ─────────────────────────
if ticker:
    with st.spinner("Crunching option chain …"):
        vol_s, gk_s = volatility_solver(ticker, rfr, side, sigma_guess, tol)
        if vol_s is not None and not vol_s.empty:
            st.success("Done!")
            plot_surface(vol_s, gk_s, greek, side, ticker)
        else:
            st.error("Failed to build surface.")
else:
    st.warning("Enter a ticker to begin.")
