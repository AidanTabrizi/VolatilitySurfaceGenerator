import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import fsolve
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from mpl_toolkits.mplot3d import Axes3D

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
def get_recent_market_day(today):
    # Use the NYSE calendar; you can adjust to other exchanges if needed
    nyse = mcal.get_calendar('NYSE')

    # Get a range of trading days around today's date
    schedule = nyse.valid_days(start_date=today - timedelta(days=7), end_date=today)

    # Find the most recent valid trading day before or equal to today
    recent_market_day = max(schedule[schedule <= today])

    return recent_market_day

# Function to calculate the implied volatility surface
def volatility_solver(ticker, rfr, option_type, sigma, tolerance):
    # Get the current date and find the most recent market day
    today = datetime.now().date()
    recent_market_day = get_recent_market_day(today)
    start_date = recent_market_day.strftime('%Y-%m-%d')

    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date)
    if stock_data.empty:
        st.error(f"No stock data available for {ticker} on {start_date}.")
        return None
    S0 = stock_data['Close'].iloc[-1]

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
            implied_volatility_df.append(implied_volatility_scalar)
        except Exception as e:
            st.error(f"Error calculating implied volatility for strike {K} and expiry {index[0]}: {e}")
            implied_volatility_df.append(np.nan)

    # Convert implied volatility list to a DataFrame with the original index
    implied_volatility_df_indexed = pd.Series(implied_volatility_df, index=df_option_data[option_type].index)

    # Interpolate missing values
    df_interpolated = implied_volatility_df_indexed.unstack(0).interpolate(method='linear')

    return df_interpolated


# Function to plot the implied volatility surface
def plot_implied_volatility_surface(vol_surface):
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


    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='#657383', linewidth=0.02, antialiased=False)


    # Add labels and title
    ax.set_xlabel('Time to Expiry (Days)', weight = 'bold')
    ax.set_ylabel('Strike Price (USD)', weight = 'bold')
    ax.set_zlabel('Implied Volatility', weight = 'bold')
    ax.set_title(f'Volatility Surface for {ticker.upper()} {option_type.capitalize()} Options', weight='bold', size ='20')
    # Add a color bar
    # Create the color bar (assuming 'surf' is your surface plot object)
    color_bar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Adjust 'shrink' and 'aspect' to fit your layout

    # Add a label to the color bar
    color_bar.set_label('Implied Volatility', color='#FFFFFF', fontsize=12, labelpad=15, weight='bold')

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
            implied_vol_surface = volatility_solver(ticker, risk_free_rate, option_type, sigma, tolerance)
            if implied_vol_surface is not None and not implied_vol_surface.empty:
                st.success('Calculation complete!')
                plot_implied_volatility_surface(implied_vol_surface)
            else:
                st.error("Failed to calculate implied volatility surface.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.warning("Please provide a valid ticker symbol and option type.")
