import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import seaborn as sns
from datetime import date

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Trading Strategy Dashboard", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Trading Strategy Analysis Dashboard")

# Define the get_daily_data_with_signals function
def get_daily_data_with_signals(symbol, start_date, end_date):
    # Download daily historical data using yfinance
    df = yf.download(symbol, start=start_date, end=end_date, interval="1d")

    # Create columns for long and short signals
    df['long_signal'] = ""
    df['short_signal'] = ""

    # Iterate through the data to add signals
    for i in range(1, len(df)):
        current_date = df.index[i]
        prev_date = df.index[i - 1]

        # Check if it's the first day of a new month
        if current_date.month != prev_date.month:
            # Add a long signal to the 'long_signal' column
            df.loc[current_date, 'long_signal'] = "long"
            # Add a short signal to the 'short_signal' column
            df.loc[current_date, 'short_signal'] = "Short"

    df = df.drop(columns=["Volume", "Adj Close"])
    return df

def Monthly_avg():
    
    max_date = date.today()
    
    # Create the input widgets with min_value and max_value
    symbol = st.text_input("Enter Symbol (e.g., BTC-USD):")
    start_date = st.date_input("Start Date", max_value=max_date)
    end_date = st.date_input("End Date", max_value=max_date)
    
    # Use the get_daily_data_with_signals function to get daily data with signals
    df = get_daily_data_with_signals(symbol, start_date, end_date)
    df.to_csv("data.csv")
    leverage = st.slider("Leverage (x)", min_value=1, max_value=10, value=1)
    initial_portfolio_val = st.number_input("Initial Portfolio Value", min_value=100)

    # Create input elements for tp (take profit) and sl (stop loss) for long and short
    long_tp_percentage = st.number_input("Take Profit for long trades (%)", min_value=0.01, max_value=1.0)
    short_tp_percentage = st.number_input("Take profit for short trades (%)", min_value=0.01, max_value=1.0)
    long_sl_percentage = st.number_input("Stop Loss for long trades (%)", min_value=0.01, max_value=1.0)
    short_sl_percentage = st.number_input("Stop Loss for short trades (%)", min_value=0.01, max_value=1.0)

    # Initialize variables to track trade performance and portfolio value
    total_trades = 0
    long_wins = 0
    short_wins = 0
    long_loose = 0
    short_loose = 0
    trades = []
    portfolio_value = initial_portfolio_val  # Initial portfolio value
    portfolio_values = [portfolio_value]  # List to track portfolio values

    for i in range(1, len(df)):
        bars_to_tp = 0  # Initialize bars_to_tp to 0
        bars_to_sl = 0  # Initialize bars_to_sl to 0
        # price_change_percentage = (current_close - open_price) / open_price
        
        if df["long_signal"].iloc[i] == "long":
            # Enter a Long trade
            total_trades +=1
            open_price = df["Close"].iloc[i - 1]
            entry_price = open_price
            stop_loss_long = entry_price * (1 - long_sl_percentage)
            take_profit_long = entry_price * (1 + long_tp_percentage)
            
            for j in range(i, len(df)):  # Start from the row where the Long signal was generated and iterate up to the second last row
                if df["long_signal"].iloc[j + 1] == "long":  # Check if the next signal is a Long signal
                    break  # Exit the loop if the next signal is Long
                if df["High"].iloc[j] > take_profit_long:
                    bars_to_tp = j - i + 1  # Number of bars to reach TP
                    break
                elif df["Low"].iloc[j] < stop_loss_long:
                    bars_to_sl = j - i + 1  # Number of bars to reach SL
                    break
                
            if bars_to_tp > 0:
            # Trade reached TP
                profit_long = portfolio_value - (portfolio_value * (1 - long_tp_percentage))
                profit_long *= leverage
                portfolio_value = portfolio_value + profit_long
                trades.append({"Date": df.index[i],
                            "Direction": "Long",
                            "EntryPrice": open_price,
                            "Close":take_profit_long,
                            "Status": "Win 💹",
                            "BarsToTP": bars_to_tp,
                            "portfolio_value": portfolio_value})
                long_wins += 1
            
            elif bars_to_sl > 0:
                # Trade reached SL
                long_loss_at_sl = portfolio_value - (portfolio_value * (1 - (long_sl_percentage)))
                long_loss_at_sl *= leverage
                portfolio_value = portfolio_value - (long_loss_at_sl)
                trades.append({"Date": df.index[i],
                            "Direction": "Long",
                            "EntryPrice": open_price,
                            "Close": stop_loss_long,
                            "Status": "Lose ❌",
                            "BarsToSL": bars_to_sl,
                            "portfolio_value": portfolio_value,
                            "long_sl": (long_sl_percentage * leverage) * 100
                            })
                long_loose += 1

        if df["short_signal"].iloc[i] == "Short":
            # Enter a Short trade
            total_trades += 1
            open_price = df["Close"].iloc[i - 1]
            entry_price = open_price
            stop_loss_short = entry_price * (1 + short_sl_percentage)
            take_profit_short = entry_price * (1 - short_tp_percentage)

            for j in range(i, len(df)):
                if df["short_signal"].iloc[j+1] == "Short":
                    break
                if df["Low"].iloc[j] < take_profit_short:
                    bars_to_tp = j - i + 1
                    break
                elif df["High"].iloc[j] > stop_loss_short:
                    bars_to_sl = j - i + 1
                    break
            if bars_to_sl > 0:
                short_loss = portfolio_value - (portfolio_value * (1 - (short_sl_percentage)))
                short_loss *= leverage
                portfolio_value = portfolio_value - short_loss
                trades.append({"Date": df.index[i],
                            "Direction": "Short",
                            "EntryPrice": entry_price,
                            "Close": stop_loss_short,
                            "Status": "Lose❌",
                            "BarsToSL": bars_to_sl,
                            "portfolio_value": portfolio_value,
                            "short_sl": (short_sl_percentage * leverage) * 100
                            })  
                short_loose += 1
            elif bars_to_tp > 0:
                short_wins += 1
                profit = portfolio_value - (portfolio_value * (1 - short_tp_percentage))
                profit *= leverage
                portfolio_value = portfolio_value + profit
                trades.append({"Date": df.index[i],
                            "Direction": "Short",
                            "EntryPrice": entry_price,
                            "Close": take_profit_short,
                            "Status": "Win💹",
                            "BarsToTP": bars_to_tp,
                            "portfolio_value": portfolio_value
                            })
        portfolio_values.append(portfolio_value)  # Append portfolio value to the list   

    # Calculate strategy performance
    long_win_rate = (long_wins / total_trades) * 2
    short_win_rate = (short_wins / total_trades) * 2
    loose_trades = long_loose + short_loose
    Win_trades = total_trades - loose_trades

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv("trades_2.csv", index=False)

    # st.title('Trading Strategy Analysis Dashboard')
    trades_df = pd.read_csv("trades_2.csv")

    def format_float(val):
        if isinstance(val, (int, float)):
            return f"{val:.0f}"  # Format as float with 2 decimal places
        else:
            return val

    trades_df["EntryPrice"] = trades_df["EntryPrice"].apply(format_float)
    trades_df["Close"] = trades_df["Close"].apply(format_float)

    # Display trade win and lose rates as a pie chart
    st.write("## Trade Win/Loss Rates")
    labels = ['Wins', 'Losses']
    sizes = [Win_trades, loose_trades]
    fig1, ax1 = plt.subplots(figsize=(2, 2))  # Adjust the width and height as needed

    # Set label text color to white
    custom_colors = ['#2ca02c', '#d62728']
    ax1.set_title("Trade Win/Loss Rates", color="white")
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'},
            colors=custom_colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Make the background transparent
    fig1.patch.set_facecolor('none')  # Set background color to transparent
    ax1.set_facecolor('none')  # Set subplot background color to transparent
    st.pyplot(fig1)

    # Display trading performance
    st.write("## Portfolio Growth")
    fig, ax = plt.subplots()
    ax.plot(df.index, portfolio_values)

    # Set the figure size (width, height) in inches
    fig, ax = plt.subplots(figsize=(4, 2))  # Adjust the width and height as needed
    ax.plot(df.index, portfolio_values)

    # Set title text color to white
    ax.set_title("Portfolio Growth", color="white")

    # Set tick text color to white
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Make the background transparent
    fig.patch.set_facecolor('none')  # Set background color to transparent
    ax.set_facecolor('none')  # Set subplot background color to transparent
    ax.patch.set_alpha(0)  # Set subplot background alpha (transparency) to 0

    st.pyplot(fig)

    # Display line chart of close prices over time
    st.write("## Close Prices Over Time")
    st.line_chart(df["Close"])

    # st.write("Long Trade Win Rate: {:.2%}".format(long_win_rate))
    st.metric(label="Long Trade Win Rate", value= "{:.2%}".format(long_win_rate))
    st.metric(label="Short Trade Win Rate", value= "{:.2%}".format(short_win_rate))
    st.metric(label="Lose trades rate", value=  str(loose_trades))
    st.metric(label="Win trades rate", value= str(Win_trades))
    st.metric(label="total trades", value= str(total_trades))
    st.metric(label="Overall Strategy Win Rate", value= " {:.2%}".format((long_wins + short_wins) / total_trades))
    # Display list of trades from trades_2.csv
    st.write("\nList of Trades:")
    st.write(trades_df)

    # Calculate portfolio growth
    portfolio_growth = (portfolio_value - initial_portfolio_val) / initial_portfolio_val
    st.metric(label="Portfolio Growth", value= "{:.2%}".format(portfolio_growth))
    # st.write("\nPortfolio Growth: {:.2%}".format(portfolio_growth))

    formatted_portfolio_value = "{:,.0f}".format(portfolio_value)
    st.metric(label="Current portfolio value ", value= formatted_portfolio_value)
    # st.write("Current portfolio value is: " + formatted_portfolio_value)
    
    # Bar chart for total wins and losses in long and short
    st.write("## Total Wins and Losses in Long and Short Trades")
    long_wins_total = trades_df[trades_df['Direction'] == 'Long']['Status'].str.contains('Win').sum()
    long_losses_total = trades_df[trades_df['Direction'] == 'Long']['Status'].str.contains('Lose').sum()
    short_wins_total = trades_df[trades_df['Direction'] == 'Short']['Status'].str.contains('Win').sum()
    short_losses_total = trades_df[trades_df['Direction'] == 'Short']['Status'].str.contains('Lose').sum()

    labels = ['Long Wins', 'Long Losses', 'Short Wins', 'Short Losses']
    values = [long_wins_total, long_losses_total, short_wins_total, short_losses_total]

    # Change the inside color of the bars to blue and orange
    colors = ['green', 'red', 'green', 'red']
    fig2, ax2 = plt.subplots(figsize=(6, 3))  # Adjust the width and height as needed
    bars = ax2.bar(labels, values, color=colors)

    # Set the background color of the subplot
    ax2.set_facecolor('black')  # Change 'black' to any color you prefer

    ax2.set_title("Total Wins and Losses in Long and Short Trades", color="white")
    ax2.set_ylabel("Count", color="white")
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')

    # Add number values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax2.annotate(f'{yval}', xy=(bar.get_x() + bar.get_width() / 2, yval),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', color='white')

    # Make the figure background transparent
    fig2.patch.set_alpha(0)  # Set figure background alpha (transparency) to 0

    st.pyplot(fig2)
    
selected_strategy = st.sidebar.selectbox("✅Choose a strategy", ["Strategy_(S1)","s1_with_constant_trade_size", "Monthly_avg","Monthly_close"])

# Call the selected strategy based on user input
if selected_strategy == "Strategy_(S1)":
    Monthly_avg()


