# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import date
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor


# endregion

class KANYEWEST(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 7, 25)
        self.SetEndDate(2024, 7, 25)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol # symbol has to be SPY
        self.set_benchmark(self.symbol)

        self.cap = 100000
        self.benchmark_chart = []
        
        # Initialize model with past data
        history = self.History(self.symbol, 1800, Resolution.Daily)
        self.model_inter, self.model_intra = self.train(history)

        # Track open stop market orders
        self.stopMarketTicket = None

    def OnData(self, data: Slice):
        self.plot_market()
        if not data.ContainsKey(self.symbol):
            return

        if self.model_inter is None or self.model_intra is None:
            self.Log("Models are not trained. Skipping prediction.")
            return
        
        history = self.History(self.symbol, 35, Resolution.Daily)
        formatted_data = self.prepare_data(history)
        
        if formatted_data.empty:
            self.Log("Formatted data is empty. Skipping this OnData call.")
            return

        # Get latest features
        latest_data = formatted_data.iloc[[-1]][['0_inter', '0_intra', '1_inter', '1_intra', '2_inter', '2_intra', 'SMA_10', 'SMA_20']]

        # if date.day == 15:
        #     self.liquidate()
        
        # Get predictions
        inter_pred = self.model_inter.predict(latest_data)[0]
        intra_pred = self.model_intra.predict(latest_data)[0]
        
        self.Log(f"Date: {self.Time.date()}, Interday Prediction: {inter_pred:.4f}, Intraday Prediction: {intra_pred:.4f}")

        # Cancel any existing stop market orders before making a new trade
        if self.stopMarketTicket is not None and self.stopMarketTicket.Status == OrderStatus.Submitted:
            self.Transactions.CancelOrder(self.stopMarketTicket.OrderId)
            self.stopMarketTicket = None

        # paramseters to tune
        buy_threshold = 0.5
        sell_threshold = 0.4
        short_threshold = 0.35
        portfolio_weight = 0.7
        short_weight = 0.7

        # trading logic
        if inter_pred > buy_threshold and intra_pred > buy_threshold:  # Strong bullish signal
            self.SetHoldings(self.symbol, portfolio_weight)
            purchase_price = self.Portfolio[self.symbol].Price
            self.RecordPurchase(self.Time, 0.5 * purchase_price)
            self.PlaceStopMarketOrder(purchase_price, direction="long")
        # elif inter_pred < sell_threshold and intra_pred < sell_threshold: 
        elif inter_pred < sell_threshold and inter_pred > short_threshold and intra_pred < buy_threshold and intra_pred > short_threshold: 
            pass
        elif inter_pred < short_threshold and intra_pred < short_threshold: # Strong bearish signal
            self.SetHoldings(self.symbol, -short_weight) #portfolio_weight)
            purchase_price = self.Portfolio[self.symbol].Price
            self.RecordPurchase(self.Time, 0.5 * purchase_price)
            self.PlaceStopMarketOrder(purchase_price, direction="short")

        

    def PlaceStopMarketOrder(self, purchase_price, direction):
        if direction == "long":
            stop_price = purchase_price * 0.99  # 1% below purchase price
            self.stopMarketTicket = self.StopMarketOrder(self.symbol, -self.Portfolio[self.symbol].Quantity, stop_price)
            self.Log(f"Placed stop market order to sell at ${stop_price:.2f} (0.5% below purchase price).")
        elif direction == "short":
            stop_price = purchase_price * 1.01  # 1% above purchase price
            self.stopMarketTicket = self.StopMarketOrder(self.symbol, -self.Portfolio[self.symbol].Quantity, stop_price)
            self.Log(f"Placed stop market order to cover at ${stop_price:.2f} (0.5% above purchase price).")

    def RecordPurchase(self, date, amount):
        self.Log(f"Purchase recorded: Date: {date}, Amount: ${amount:.2f}")

    def OnOrderEvent(self, orderEvent):
        # Check if the stop market order was filled
        if self.stopMarketTicket is not None and orderEvent.OrderId == self.stopMarketTicket.OrderId:
            if orderEvent.Status == OrderStatus.Filled:
                execution_price = orderEvent.FillPrice
                execution_date = self.Time
                self.Log(f"Stop market order executed: Date: {execution_date}, Price: ${execution_price:.2f}")
                # Clear the stop market ticket after execution
                self.stopMarketTicket = None


        def train(self, history):
            """
            Train interday and intraday predictive models using historical data.

            Args:
                history: Historical data to train the models.

            Returns:
                model_inter: Trained model for interday predictions.
                model_intra: Trained model for intraday predictions.
            """
            # Convert history to a DataFrame
            df = pd.DataFrame(history)
            grads = self.prepare_data(df)

            # Check if the prepared data is empty
            if grads.empty:
                self.Log("Training data is empty. Cannot train model.")
                return None, None

            # Prepare features and targets
            X = grads[['0_inter', '0_intra', '1_inter', '1_intra', '2_inter', '2_intra', 'SMA_10', 'SMA_20']]
            y_inter = grads['0_inter'].shift(1)[grads.index]
            y_intra = grads['0_intra'].shift(1)[grads.index]

            # Remove the first 20 rows to ensure valid data
            X = X[20:]
            y_inter = y_inter[20:]
            y_intra = y_intra[20:]

            # Train XGBoost models for interday and intraday predictions
            self.model_inter = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
            self.model_intra = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)

            self.model_inter.fit(X, y_inter)
            self.model_intra.fit(X, y_intra)

            self.Log("Models trained successfully with XGBoost.")
            
            return (self.model_inter, self.model_intra)

    def prepare_data(self, df):
        df = df.reset_index()
        if 'index' in df.columns:
            df.rename(columns={'index': 'time'}, inplace=True)
        elif 'time' not in df.columns:
            df.rename(columns={'time': 'time'}, inplace=True)

        """

        Implement your own feature engineering and data preparation here. 


        """

        df['Date'] = pd.to_datetime(df['time'])
        df.set_index('Date', inplace=True)
        
        df['intraday_grads'] = (df['close'] / df['open'] - 1).dropna()
        df['interday_grads'] = (df['open'] / df['close'].shift(1) - 1).dropna()
        df['intraday_grads_norm'] = (df['intraday_grads'] - df['intraday_grads'].min()) / (df['intraday_grads'].max() - df['intraday_grads'].min())
        df['interday_grads_norm'] = (df['interday_grads'] - df['interday_grads'].min()) / (df['interday_grads'].max() - df['interday_grads'].min())
        
        grads = df[['interday_grads_norm', 'intraday_grads_norm']]
        grads.columns = ['0_inter', '0_intra']
        grads['1_inter'] = df['interday_grads_norm'].shift(-1)
        grads['1_intra'] = df['intraday_grads_norm'].shift(-1)
        grads['2_inter'] = df['interday_grads_norm'].shift(-2)
        grads['2_intra'] = df['intraday_grads_norm'].shift(-2)
        grads['SMA_10'] = df['close'].rolling(window=10).mean()  # 10-day Simple Moving Average
        grads['SMA_20'] = df['close'].rolling(window=20).mean()
        
        return grads.dropna()



    def plot_market(self): #plot the market on the Startegy Equity Chart with your portfolio
        hist = self.History([self.symbol], 252, Resolution.Daily)['close'].unstack(level=0).dropna()
        self.benchmark_chart.append(hist[self.symbol].iloc[-1])
        benchmark_perf = self.benchmark_chart[-1] / self.benchmark_chart[0] * self.cap
        self.Plot("Strategy Equity", "Buy & Hold", benchmark_perf)