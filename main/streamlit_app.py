import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

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


# Function to find the most recent market day if today is a weekend
def get_recent_market_day():
    # Use the current date in US Eastern Time

    today = datetime.today().date()
    # If today is Saturday, go back to Friday
    if today.weekday() == 5:  # Saturday
        return today - timedelta(days=1)
    # If today is Sunday, go back to Friday
    elif today.weekday() == 6:  # Sunday
        return today - timedelta(days=2)
    else:
        return today

# Function to calculate the implied volatility surface
def volatility_solver(ticker, rfr, option_type, sigma, tolerance):
    # Get the current date and find the most recent market day
    recent_market_day = get_recent_market_day()
    start_date = recent_market_day.strftime('%Y-%m-%d')

    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date)
    if stock_data.empty:
        st.error(f"No stock data available for {ticker} on {start_date}.")
        return None
    S0 = stock_data['Close'].iloc[-1]
    today = stock_data.index[-1]  # Update 'today' to be the last date in stock_data

    # Fetch option data
    ticker_info = yf.Ticker(ticker)
    option_data = ticker_info.options
    df_option_data = []

    # Collect options data for each expiration date
    for expiration_date in option_data:
        option_chain = ticker_info.option_chain(expiration_date)
        calls = option_chain.calls
        puts = option_chain.puts

        # Add call option data
        for _, option in calls.iterrows():
            strike = option['strike']
            midprice = (option['bid'] + option['ask']) / 2
            df_option_data.append([expiration_date, strike, midprice, 'CALL'])

        # Add put option data
        for _, option in puts.iterrows():
            strike = option['strike']
            midprice = (option['bid'] + option['ask']) / 2
            df_option_data.append([expiration_date, strike, midprice, 'PUT'])

    # Convert the list to a DataFrame
    df_option_data = pd.DataFrame(df_option_data, columns=['expiration_date', 'strike', 'midprice', 'type'])

    # Filter the data to strikes within 20% of the current stock price
    df_option_data = df_option_data[(S0 * 0.8 < df_option_data['strike']) & (df_option_data['strike'] < S0 * 1.2)]

    # Calculate days to expiry using 'today' as the most recent market day
    df_option_data['days_to_expiry'] = pd.to_datetime(df_option_data['expiration_date'])
    df_option_data['expiration_date'] = (df_option_data['days_to_expiry'] - today).dt.days

    # Filter by expiry dates within 100 days
    df_option_data = df_option_data[df_option_data['expiration_date'] > 0]
    df_option_data = df_option_data[df_option_data['expiration_date'] < 100]

    # Set index and pivot the table for easier access
    df_option_data = df_option_data.set_index(['expiration_date', 'strike', 'type']).sort_index()
    df_option_data = df_option_data.pivot_table(index=['expiration_date', 'strike'], columns='type', values='midprice')

    # Black-Scholes model function for implied volatility calculation
    def BlackScholesModel(sigma, S0, K, P, T, r, option_type):
        d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'CALL':
            return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - P
        elif option_type == 'PUT':
            return K * np.exp(-r * T) * norm.cdf(-d2) - P - S0 * norm.cdf(-d1)

    # List to hold implied volatilities
    implied_volatility_df = []
    def calculate_greeks(S0, K, T, r, sigma, option_type):
        d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'CALL':
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        elif option_type == 'PUT':
            delta = norm.cdf(d1) - 1
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
        theta = - (S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'CALL' else norm.cdf(-d2)))
        theta = theta / 365
        vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100


        return delta, gamma, theta, vega, rho

    greeks_df = []

    # Calculate implied volatility for each option
    for index, value in df_option_data[option_type].items():
        T = index[0] / 365
        K = index[1]
        sigma0 = sigma
        try:
            implied_volatility = fsolve(
                BlackScholesModel, sigma0,
                args=(S0, K, value, T, rfr, option_type),
                xtol=tolerance
            )
            implied_volatility_scalar = float(implied_volatility[0])
            delta, gamma, theta, vega, rho = calculate_greeks(S0, K, T, rfr, implied_volatility_scalar, option_type)
            if 0 < implied_volatility_scalar < 5:
                implied_volatility_df.append(implied_volatility_scalar)
                greeks_df.append([delta, gamma, theta, vega, rho])
            else:
                implied_volatility_df.append(np.nan)
                greeks_df.append([np.nan, np.nan, np.nan, np.nan, np.nan])
        except Exception as e:
            st.error(f"Error calculating implied volatility for strike {K} and expiry {index[0]}: {e}")
            implied_volatility_df.append(np.nan)
            greeks_df.append([np.nan, np.nan, np.nan, np.nan, np.nan])

    # Convert implied volatility list to a DataFrame with the original index
    implied_volatility_df_indexed = pd.Series(implied_volatility_df, index=df_option_data[option_type].index)
    greeks_df_indexed = pd.DataFrame(greeks_df, index=df_option_data[option_type].index, columns=['DELTA', 'GAMMA', 'THETA', 'VEGA', 'RHO'])

    # Interpolate missing values
    df_interpolated = implied_volatility_df_indexed.unstack(0).interpolate(method='linear')
    greeks_interpolated = greeks_df_indexed.unstack(0).interpolate(method='linear')

    return df_interpolated, greeks_interpolated


# Function to plot the implied volatility surface
def plot_implied_volatility_surface(vol_surface, greek_surface, greek_parameter):
    custom_style = {
        'axes.facecolor': '#000000',  # Background color of the plot
        'axes.edgecolor': '#FFFFFF',  # Edge color of the plot
        'axes.labelcolor': '#FFFFFF',  # Color of x, y, z axis labels
        'figure.facecolor': '#000000',  # Background color of the figure
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

    # Prepare data for plotting
    X = vol_surface.columns.values  # Expiry times
    Y = vol_surface.index.values    # Strike prices
    X, Y = np.meshgrid(X, Y)
    Z = vol_surface.values
    C = greek_surface[greek_parameter].values

    # Set appropriate normalization based on the Greek parameter
    if greek_parameter == 'DELTA':
        if option_type == 'PUT':
            norm = Normalize(vmin=-1, vmax=0)
            cmap = 'plasma_r'
        else:
            norm = Normalize(vmin=0, vmax=1)
            cmap = 'plasma'
    elif greek_parameter == 'GAMMA':
        norm = Normalize(vmin=min(C.min(), 0), vmax=max(C.max(), 0.2))
        cmap = 'plasma'
    elif greek_parameter == 'THETA':
        norm = Normalize(vmin=min(C.min(), -0.5), vmax=max(C.max(), 0))
        cmap = 'plasma_r'
    elif greek_parameter == 'VEGA':
        norm = Normalize(vmin=min(C.min(), 0), vmax=max(C.max(), 0.5))
        cmap = 'plasma'
    elif greek_parameter == 'RHO':
        if option_type == 'PUT':
            norm = Normalize(vmin=-0.5, vmax=0)
            cmap = 'plasma_r'
        else:
            norm = Normalize(vmin=0, vmax=0.5)
            cmap = 'plasma'
    # Create the figure and axes
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


    colormap = plt.colormaps.get_cmap(cmap)

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, facecolors=colormap(norm(C)), rstride=1,cstride=1, edgecolor='#657383', linewidth=0.02, antialiased=False)


    # Add labels and title
    ax.set_xlabel('Time to Expiry (Days)', weight = 'bold')
    ax.set_ylabel('Strike Price (USD)', weight = 'bold')
    ax.set_zlabel('Implied Volatility', weight = 'bold')
    ax.set_title(f'Volatility Surface with {greek_parameter.capitalize()} for {ticker.upper()} {option_type.capitalize()} Options', weight='bold', size ='20')

    # Add a color bar
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(C)
    color_bar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)  # Adjust 'shrink' and 'aspect' to fit your layout

    # Add a label to the color bar
    color_bar.set_label(f'{greek_parameter.capitalize()}', color='#FFFFFF', fontsize=12, labelpad=15, weight='bold')

    # Set the tick parameters (optional customization)
    color_bar.ax.tick_params(labelsize=10, labelcolor='#FFFFFF')



    # Display the plot in Streamlit
    st.pyplot(fig)

# Streamlit UI
with st.sidebar:
    st.title("Black Scholes Volatility Surface Generator")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/aidan-tabrizi/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Aidan Tabrizi`</a>', unsafe_allow_html=True)

# Input fields in the sidebar
ticker = st.sidebar.text_input('Ticker Symbol:', value='AAPL')
option_type = st.sidebar.selectbox('Option Type:', ['CALL', 'PUT'])
greek_parameter = st.sidebar.selectbox('Heatmap Parameter:', ['DELTA','GAMMA','THETA','VEGA','RHO'])
risk_free_rate = st.sidebar.number_input("Risk-Free Rate:", value=0.04)
# Added input field for initial volatility guess
sigma = st.sidebar.number_input("Initial Volatility Guess:", value=0.4, step=0.01)
tolerance = 1e-9  # Tolerance for the solver
st.sidebar.write("Visualize the volatility surface for a call or put option of any chosen security! Just enter the ticker symbol, select the option type, input the risk-free rate, and provide an initial guess for volatility. Using market option prices from Yahoo Finance, the implied volatility is calculated with the Black Scholes model and plotted via Matplotlib.")

# Main Content


# Generate and plot the implied volatility surface if inputs are valid
if ticker and option_type:
    with st.spinner('Calculating implied volatility surface...'):
        try:
            implied_vol_surface, greeks_surface = volatility_solver(ticker, risk_free_rate, option_type, sigma, tolerance)
            if implied_vol_surface is not None and not implied_vol_surface.empty:
                st.success('Calculation complete!')
                plot_implied_volatility_surface(implied_vol_surface, greeks_surface, greek_parameter)
            else:
                st.error("Failed to calculate implied volatility surface.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.warning("Please provide a valid ticker symbol and option type.")
