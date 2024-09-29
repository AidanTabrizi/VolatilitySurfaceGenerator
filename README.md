# Volatility Surface Generator

This project is a web application designed to plot the implied volatility surface of put and call options for a given security. By entering a ticker symbol and other necessary parameters, users can visualize the volatility surface using the Black-Scholes model (constant volatility) along with the corresponding Greeks.

## Key Features:
1. **Dynamic Data Fetching**: Uses the yFinance API to fetch real-time market data, including option prices, historical data, and other necessary inputs.

2. **Volatility Modeling**: Utilizes the Black-Scholes model, which assumes constant volatility, to generate the implied volatility surface based on user inputs.

3. **Greek Visualization**: Provides visualiation of key option Greeks (Delta, Gamma, Theta, Vega, and Rho) directly on the heatmap overlaid on the volatility surface. 

4. **3D Interactive Plots**: Generates 3D plots of implied volatility surface, allowing users to explore changes in implied volatility relative to strike price and time to maturity, enriched with colour-coded Greek values.
   
5. **User-Friendly Interface**: A streamlined web interface where users can input a ticker, set the risk-free rate, initial guess for volatility, desired Greek for visualization, choose the type of option, and instantly see the results.



## Requirements:
- `streamlit` to create the web application interface and handle user inputs
- `yfinance` to fetch real-time market data, including stock prices and option chains
- `numpy` for numerical calculations and data manipulation
- `pandas` for organizing and manipulating data, especially for handling option data in tabular form
- `matplotlib` to create and display 3D plots of the implied volatility surface and heatmaps of Greeks
- `datetime` to handle dates and calculate days to expiration for options
- `scipy` to solve for implied volatility and to access cumulative/probability density functions
