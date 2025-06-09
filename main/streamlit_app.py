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
        header, footer, .css-1y4p8pa {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────  Helper functions  ────────────────────────
def fetch_stock_data(ticker: str, ref_date: datetime.date, max_tries: int = 5):
    """Download most-recent price data, stepping back a day if a holiday/weekend."""
    for _ in range(max_tries):
        df = yf.download(ticker, start=ref_date)
        if not df.empty:
            return df
        ref_date -= timedelta(days=1)
    return None


def recent_market_day() -> datetime.date:
    today = datetime.today().date()
    if today.weekday() == 5:                       # Saturday → Friday
        return today - timedelta(days=1)
    if today.weekday() == 6:                       # Sunday   → Friday
        return today - timedelta(days=2)
    return today

# ─────────────────────────  Core IV-surface solver  ────────────────────
def volatility_solver(ticker: str,
                      rfr: float,
                      opt_type: str,
                      sigma0: float,
                      tol: float):
    # 1) Spot price -----------------------------------------------------
    trade_date = recent_market_day()
    spot_df = fetch_stock_data(ticker, trade_date)
    if spot_df is None or spot_df.empty:
        st.error("Couldn’t fetch recent price data.")
        return None, None
    S0 = float(spot_df["Close"].iloc[-1])
    today = spot_df.index[-1]

    # 2) Option chain ---------------------------------------------------
    rows = []
    tkr = yf.Ticker(ticker)
    for exp in tkr.options:
        chain = tkr.option_chain(exp)
        for _, row in chain.calls.iterrows():
            rows.append([exp, row["strike"], (row["bid"] + row["ask"]) / 2, "CALL"])
        for _, row in chain.puts.iterrows():
            rows.append([exp, row["strike"], (row["bid"] + row["ask"]) / 2, "PUT"])

    if not rows:
        st.error("No option data returned from Yahoo.")
        return None, None

    df = pd.DataFrame(rows, columns=["expiry", "strike", "midprice", "type"])
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df.dropna(subset=["strike", "midprice"], inplace=True)
    df.drop_duplicates(subset=["expiry", "strike", "type"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3) Keep strikes ±20 % of spot  (NumPy masks avoid index alignment) -
    strikes = df["strike"].to_numpy(dtype=float)
    mask_strike = (strikes > 0.8 * S0) & (strikes < 1.2 * S0)
    df = df.iloc[mask_strike]

    # 4) Convert expiry → days-to-expiry and keep 1–99 days ------------
    df["dte"] = (pd.to_datetime(df["expiry"]) - today).dt.days
    dte_arr = df["dte"].to_numpy(dtype=int)
    mask_dte = (dte_arr > 0) & (dte_arr < 100)
    df = df.iloc[mask_dte]

    if df.empty:
        st.error("No options inside strike/DTE filters.")
        return None, None

    # 5) Pivot to (dte, strike) index × CALL/PUT columns ---------------
    df = (
        df.pivot_table(index=["dte", "strike"], columns="type", values="midprice")
          .sort_index()
    )
    if opt_type not in df.columns:
        st.error(f"No {opt_type.lower()} prices after filtering.")
        return None, None

    # 6) Black-Scholes + Greeks  ---------------------------------------
    def bs_residual(sig, S, K, P, T, r, otype):
        d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        if otype == "CALL":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - P
        return K * np.exp(-r * T) * norm.cdf(-d2) - P - S * norm.cdf(-d1)

    iv_list, greek_list = [], []
    for (dte, K), price in df[opt_type].items():
        T = dte / 365.0
        try:
            iv = float(
                fsolve(
                    bs_residual,
                    sigma0,
                    args=(S0, K, price, T, rfr, opt_type),
                    xtol=tol,
                )[0]
            )
        except Exception:  # fallback if fsolve fails
            iv = np.nan

        # Greeks (only if iv solved)
        if np.isnan(iv) or iv <= 0:
            iv_list.append(np.nan)
            greek_list.append([np.nan] * 5)
            continue

        d1 = (np.log(S0 / K) + (rfr + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)

        if opt_type == "CALL":
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-rfr * T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            rho = -K * T * np.exp(-rfr * T) * norm.cdf(-d2) / 100

        gamma = norm.pdf(d1) / (S0 * iv * np.sqrt(T))
        theta = (
            -(
                S0 * norm.pdf(d1) * iv / (2 * np.sqrt(T))
                - rfr
                * K
                * np.exp(-rfr * T)
                * (norm.cdf(d2) if opt_type == "CALL" else norm.cdf(-d2))
            )
            / 365
        )
        vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100

        iv_list.append(iv)
        greek_list.append([delta, gamma, theta, vega, rho])

    iv_series = pd.Series(iv_list, index=df[opt_type].index)
    greek_df = pd.DataFrame(
        greek_list,
        index=df[opt_type].index,
        columns=["DELTA", "GAMMA", "THETA", "VEGA", "RHO"],
    )

    iv_grid = iv_series.unstack(level=0).interpolate("linear")
    greek_grid = greek_df.unstack(level=0).interpolate("linear").fillna(0)
    return iv_grid, greek_grid

# ─────────────────────────────  Surface plot  ──────────────────────────
def plot_surface(vol_grid: pd.DataFrame,
                 greek_grid: pd.DataFrame,
                 greek: str,
                 ticker: str,
                 opt_type: str):
    X = vol_grid.columns.values               # days-to-exp (meshgrid x-axis)
    Y = vol_grid.index.values                 # strikes   (meshgrid y-axis)
    X, Y = np.meshgrid(X, Y)
    Z = vol_grid.values
    C = greek_grid[greek].values

    # Colour map & normalisation by Greek
    if greek == "DELTA":
        cmap = "plasma_r" if opt_type == "PUT" else "plasma"
        norm = Normalize(-1 if opt_type == "PUT" else 0, 0 if opt_type == "PUT" else 1)
    elif greek == "GAMMA":
        cmap = "plasma";  norm = Normalize(0, np.nanmax(C))
    elif greek == "THETA":
        cmap = "plasma_r"; norm = Normalize(np.nanmin(C), 0)
    elif greek == "VEGA":
        cmap = "plasma";  norm = Normalize(0, np.nanmax(C))
    else:  # RHO
        cmap = "plasma_r" if opt_type == "PUT" else "plasma"
        norm = Normalize(np.nanmin(C), np.nanmax(C))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z,
                    facecolors=plt.cm.get_cmap(cmap)(norm(C)),
                    linewidth=0, antialiased=False)
    ax.set_xlabel("Days to Expiry")
    ax.set_ylabel("Strike")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"{ticker.upper()} {opt_type} IV Surface • coloured by {greek}")
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm); m.set_array(C)
    fig.colorbar(m, shrink=0.5, aspect=8, label=greek)
    st.pyplot(fig)

# ─────────────────────────────  Streamlit UI  ──────────────────────────
with st.sidebar:
    st.title("Volatility Surface Generator")
    ticker      = st.text_input("Ticker", value="AAPL")
    opt_type    = st.selectbox("Option Type", ["CALL", "PUT"])
    greek       = st.selectbox("Colour by Greek", ["DELTA", "GAMMA", "THETA", "VEGA", "RHO"])
    rfr         = st.number_input("Risk-free rate", value=0.04)
    sigma_guess = st.number_input("Initial vol guess", value=0.4, step=0.01)
    tol         = 1e-9

if ticker:
    with st.spinner("Computing IV surface …"):
        iv_grid, gk_grid = volatility_solver(ticker, rfr, opt_type, sigma_guess, tol)

    if iv_grid is not None and not iv_grid.empty:
        st.success("Calculation complete!")
        plot_surface(iv_grid, gk_grid, greek, ticker, opt_type)
    else:
        st.error("Failed to compute surface — check filters or data availability.")
