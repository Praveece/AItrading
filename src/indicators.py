import pandas as pd
import numpy as np
import logging

def calculate_indicators(df):
    df = df.copy()
    try:
        # Standardize column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.capitalize()

        # Validate required columns
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logging.error(f"DataFrame missing required columns: {missing}")
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Drop rows with NaNs in required columns
        original_len = len(df)
        df = df.dropna(subset=required_cols)
        dropped = original_len - len(df)
        if dropped > 0:
            logging.warning(f"Dropped {dropped} rows due to NaNs in required columns.")

        # Basic Moving Averages & Bands
        df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['STD20'] = df['Close'].rolling(window=20, min_periods=1).std()
        df['Upper Band'] = df['SMA20'] + 2 * df['STD20']
        df['Lower Band'] = df['SMA20'] - 2 * df['STD20']

        # Exponential Moving Averages
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # RSI (14)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)  # Avoid division by zero
        df['RSI14'] = 100 - (100 / (1 + rs))
        df['RSI14'] = df['RSI14'].fillna(50)  # Neutral value for early data

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # ATR (14)
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift(1)).abs()
        lc = (df['Low'] - df['Close'].shift(1)).abs()
        tr = np.maximum(np.maximum(hl, hc), lc)
        df['ATR14'] = tr.rolling(window=14, min_periods=1).mean()

        # Volume-based indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

        # Volume surge
        vol_ma = df['Volume'].rolling(window=20, min_periods=1).mean().replace(0, 1e-9)
        df['Vol_MA20'] = vol_ma
        df['VolSurge'] = df['Volume'] / df['Vol_MA20']
        df['VolSurge'] = df['VolSurge'].fillna(1)  # Neutral value

        # Stochastic Oscillator
        low14 = df['Low'].rolling(window=14, min_periods=1).min()
        high14 = df['High'].rolling(window=14, min_periods=1).max()
        df['Stoch%K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)
        df['Stoch%D'] = df['Stoch%K'].rolling(window=3, min_periods=1).mean()

        # CCI (20)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI20'] = (tp - tp.rolling(window=20, min_periods=1).mean()) / \
                      (0.015 * tp.rolling(window=20, min_periods=1).std())

        # Money Flow Index (14)
        mf = tp * df['Volume']
        pos_flow = mf.where(tp > tp.shift(1), 0).rolling(window=14, min_periods=1).sum()
        neg_flow = mf.where(tp < tp.shift(1), 0).rolling(window=14, min_periods=1).sum().replace(0, 1e-9)
        df['MFI14'] = 100 - (100 / (1 + pos_flow / neg_flow))

        # Chaikin Money Flow (20)
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / \
              ((df['High'] - df['Low']).replace(0, np.nan))
        mfv = mfm * df['Volume']
        vol_sum = df['Volume'].rolling(window=20, min_periods=1).sum().replace(0, 1e-9)
        df['CMF20'] = mfv.rolling(window=20, min_periods=1).sum() / vol_sum
        df['CMF20'] = df['CMF20'].fillna(0)  # Neutral value

        # Support & Resistance (20)
        df['Res20'] = df['High'].rolling(window=20, min_periods=1).max()
        df['Sup20'] = df['Low'].rolling(window=20, min_periods=1).min()
        df['Price_to_Res'] = (df['Res20'] - df['Close']) / df['Close']

        # Log columns with NaNs before final dropna
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            logging.warning(f"NaNs detected in columns: {nan_cols}")

        return df.dropna()

    except Exception as e:
        logging.error(f"Indicator calculation error: {e}")
        raise


# # indicators.py
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier


# def calculate_indicators(df):
#     df = df.copy()
#     # Flatten MultiIndex columns (from yfinance auto_adjust output)
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)

#     # Basic price data validation
#     required_cols = {'Open','High','Low','Close','Volume'}
#     if not required_cols.issubset(df.columns):
#         raise ValueError(f"DataFrame missing required columns: {required_cols - set(df.columns)}")

#     # Basic Moving Averages & Bands
#     df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
#     df['STD20'] = df['Close'].rolling(window=20, min_periods=1).std()
#     df['Upper Band'] = df['SMA20'] + 2 * df['STD20']
#     df['Lower Band'] = df['SMA20'] - 2 * df['STD20']

#     # Exponential Moving Averages
#     df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
#     df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

#     # RSI (14)
#     delta = df['Close'].diff()
#     gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
#     loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
#     rs = gain / loss.replace(0, np.inf)
#     df['RSI14'] = 100 - (100 / (1 + rs))

#     # MACD & signal
#     ema12 = df['Close'].ewm(span=12, adjust=False).mean()
#     ema26 = df['Close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = ema12 - ema26
#     df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
#     df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

#     # ATR (14)
#     hl = df['High'] - df['Low']
#     hc = (df['High'] - df['Close'].shift(1)).abs()
#     lc = (df['Low'] - df['Close'].shift(1)).abs()
#     tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
#     df['ATR14'] = tr.rolling(window=14, min_periods=1).mean()

#     # Volume-based indicators
#     df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
#     tp = (df['High'] + df['Low'] + df['Close']) / 3
#     df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

#     # Volume surge: ensure Series
#     vol_ma = df['Volume'].rolling(window=20, min_periods=1).mean().replace(0, 1)
#     df['Vol_MA20'] = vol_ma
#     df['VolSurge'] = df['Volume'] / df['Vol_MA20']

#     # Stochastic Oscillator
#     low14 = df['Low'].rolling(window=14, min_periods=1).min()
#     high14 = df['High'].rolling(window=14, min_periods=1).max()
#     df['Stoch%K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)
#     df['Stoch%D'] = df['Stoch%K'].rolling(window=3, min_periods=1).mean()

#     # CCI (20)
#     tp = (df['High'] + df['Low'] + df['Close']) / 3
#     df['CCI20'] = (tp - tp.rolling(window=20, min_periods=1).mean()) / (0.015 * tp.rolling(window=20, min_periods=1).std())

#     # Money Flow Index (14)
#     mf = tp * df['Volume']
#     pos_flow = mf.where(tp > tp.shift(1), 0).rolling(window=14, min_periods=1).sum()
#     neg_flow = mf.where(tp < tp.shift(1), 0).rolling(window=14, min_periods=1).sum().replace(0, 1)
#     df['MFI14'] = 100 - (100 / (1 + pos_flow / neg_flow))

#     # Chaikin Money Flow (20)
#     mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / ((df['High'] - df['Low']).replace(0, np.nan))
#     mfv = mfm * df['Volume']
#     df['CMF20'] = mfv.rolling(window=20, min_periods=1).sum() / df['Volume'].rolling(window=20, min_periods=1).sum()

#     # Support & Resistance (20)
#     df['Res20'] = df['High'].rolling(window=20, min_periods=1).max()
#     df['Sup20'] = df['Low'].rolling(window=20, min_periods=1).min()
#     df['Price_to_Res'] = (df['Res20'] - df['Close']) / df['Close']

#     return df.dropna()


# # indicators.py
# import pandas as pd
# import numpy as np

# def calculate_indicators(df):
#     df = df.copy()
#     # Flatten MultiIndex columns (from yfinance auto_adjust output)
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)

#     # Basic price data validation
#     required_cols = {'Open','High','Low','Close','Volume'}
#     if not required_cols.issubset(df.columns):
#         raise ValueError(f"DataFrame missing required columns: {required_cols - set(df.columns)}")

#     # Basic Moving Averages & Bands
#     df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
#     df['STD20'] = df['Close'].rolling(window=20, min_periods=1).std()
#     df['Upper Band'] = df['SMA20'] + 2 * df['STD20']
#     df['Lower Band'] = df['SMA20'] - 2 * df['STD20']

#     # Exponential Moving Averages
#     df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
#     df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

#     # RSI (14)
#     delta = df['Close'].diff()
#     gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
#     loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
#     rs = gain / loss.replace(0, np.inf)
#     df['RSI14'] = 100 - (100 / (1 + rs))

#     # MACD & signal
#     ema12 = df['Close'].ewm(span=12, adjust=False).mean()
#     ema26 = df['Close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = ema12 - ema26
#     df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
#     df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

#     # ATR (14)
#     hl = df['High'] - df['Low']
#     hc = (df['High'] - df['Close'].shift(1)).abs()
#     lc = (df['Low'] - df['Close'].shift(1)).abs()
#     tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
#     df['ATR14'] = tr.rolling(window=14, min_periods=1).mean()

#     # Volume-based indicators
#     df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
#     tp = (df['High'] + df['Low'] + df['Close']) / 3
#     df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

#     # Volume surge: ensure Series
#     vol_ma = df['Volume'].rolling(window=20, min_periods=1).mean().replace(0, 1)
#     df['Vol_MA20'] = vol_ma
#     df['VolSurge'] = df['Volume'] / df['Vol_MA20']

#     # Stochastic Oscillator
#     low14 = df['Low'].rolling(window=14, min_periods=1).min()
#     high14 = df['High'].rolling(window=14, min_periods=1).max()
#     df['Stoch%K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)
#     df['Stoch%D'] = df['Stoch%K'].rolling(window=3, min_periods=1).mean()

#     # CCI (20)
#     tp = (df['High'] + df['Low'] + df['Close']) / 3
#     df['CCI20'] = (tp - tp.rolling(window=20, min_periods=1).mean()) / (0.015 * tp.rolling(window=20, min_periods=1).std())

#     # Money Flow Index (14)
#     mf = tp * df['Volume']
#     pos_flow = mf.where(tp > tp.shift(1), 0).rolling(window=14, min_periods=1).sum()
#     neg_flow = mf.where(tp < tp.shift(1), 0).rolling(window=14, min_periods=1).sum().replace(0, 1)
#     df['MFI14'] = 100 - (100 / (1 + pos_flow / neg_flow))

#     # Chaikin Money Flow (20)
#     mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / ((df['High'] - df['Low']).replace(0, np.nan))
#     mfv = mfm * df['Volume']
#     df['CMF20'] = mfv.rolling(window=20, min_periods=1).sum() / df['Volume'].rolling(window=20, min_periods=1).sum()

#     # Support & Resistance (20)
#     df['Res20'] = df['High'].rolling(window=20, min_periods=1).max()
#     df['Sup20'] = df['Low'].rolling(window=20, min_periods=1).min()
#     df['Price_to_Res'] = (df['Res20'] - df['Close']) / df['Close']

#     return df.dropna()