# Volatility Surface Generator

This project is a web application designed to plot the implied volatility surface of put and call options for a given security. By entering a ticker symbol and other necessary parameters, users can visualize the volatility surface using the Black-Scholes model (constant volatility).

## Key Features:
1. **Dynamic Data Fetching**: Uses the yFinance API to fetch real-time market data, including option prices, historical data, and other necessary inputs.

2. **Volatility Modeling**: Utilizes the Black-Scholes model, which assumes constant volatility, to generate the implied volatility surface based on user inputs.

3. **Interactive Plots**: Generates 3D plots of the implied volatility surface, allowing users to analyze how implied volatility changes with strike price and time to maturity.

4. **User-Friendly Interface**: A streamlined web interface where users can input a ticker, set the risk-free rate, initial guess for volatility, choose the type of option, and instantly see the results.

5. **Educational Tool**: Serves as an educational platform for understanding the intricacies of implied volatility surfaces and the Black-Scholes pricing model.

## Requirements:
- `yfinance` to fetch market data
- `numpy` for numerical calculations
- `matplotlib` to create and display 3D plots of the implied volatility surface
