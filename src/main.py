import os
import sys
import time
import datetime
import logging
import joblib
import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import telegram
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from indicators import calculate_indicators

# Load environment
load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
if not BOT_TOKEN or not CHAT_ID:
    raise EnvironmentError("Missing Telegram credentials")
bot = telegram.Bot(token=BOT_TOKEN)

# Sync wrapper for async send_message
async def _async_send(text):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        logging.error(f"Telegram send error: {e}")

def send_message(text):
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass
    if loop and loop.is_running():
        asyncio.create_task(_async_send(text))
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_async_send(text))

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
LAST_INIT = os.path.join(BASE_DIR, 'last_init.txt')

# Parameters
PSY_MOVE_FACTOR = 1.0
FIFTEEN_MOVE_FACTOR = 0.75
PROB_THRESH = 0.75
RULE_PROB_THRESH = 0.6
FEATURES = [
    'Close', 'SMA20', 'Upper Band', 'Lower Band', 'EMA20', 'EMA50',
    'RSI14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR14', 'OBV', 'VWAP',
    'VolSurge', 'Stoch%K', 'Stoch%D', 'CCI20', 'MFI14', 'CMF20', 'Price_to_Res'
]

# Logging setup
from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler(os.path.join(BASE_DIR, 'trading_system.log'), maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
today = datetime.datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(LAST_INIT) or open(LAST_INIT).read().strip() != today:
    logging.info('Trading AI system initialized')
    send_message('✅ Trading AI system initialized')
    with open(LAST_INIT, 'w') as f:
        f.write(today)

# Model load/train
def train_model():
    last_trading_day = datetime.datetime.now()
    if last_trading_day.weekday() >= 5:
        last_trading_day -= datetime.timedelta(days=last_trading_day.weekday() - 4)
    start_date = last_trading_day - datetime.timedelta(days=7)
    logging.info(f"Fetching 1m data from {start_date.strftime('%Y-%m-%d')} to {last_trading_day.strftime('%Y-%m-%d')}")
    max_retries = 3
    base_delay = 60
    for attempt in range(max_retries):
        try:
            df = yf.download('^NSEI', start=start_date, end=last_trading_day, interval='1m', auto_adjust=True)
            if df.empty:
                raise ValueError("No data returned from yfinance")
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tz_convert('Asia/Kolkata')
            df = df.sort_index()
            break
        except Exception as e:
            logging.error(f"yfinance fetch error on attempt {attempt + 1}: {str(e)}")
            if "YFRateLimitError" in str(e):
                sleep_time = base_delay * (2 ** attempt)
                logging.info(f"Rate limit hit, retrying after {sleep_time} seconds")
                time.sleep(sleep_time)
            elif attempt == max_retries - 1:
                logging.warning("Max retries reached, using default model")
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                joblib.dump(model, MODEL_PATH)
                return model
            else:
                time.sleep(base_delay)
    if df.empty or len(df) < 100:
        logging.warning(f"Insufficient data (rows: {len(df)}). Using default model.")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        joblib.dump(model, MODEL_PATH)
        return model
    logging.info(f"Training with {len(df)} samples")
    df = calculate_indicators(df)
    fut = df.shift(-3)
    df['Breakout'] = np.where(
        fut['Close'] > fut['Upper Band'], 1,
        np.where(fut['Close'] < fut['Lower Band'], -1, 0)
    )
    df.dropna(inplace=True)
    X, y = df[FEATURES], df['Breakout']
    n_samples = len(df)
    max_splits = min(5, n_samples // 10)
    if max_splits < 2:
        logging.warning('Not enough samples for cross-validation; training without CV')
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        return model
    tscv = TimeSeriesSplit(n_splits=max_splits)
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    clf = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_distributions=param_dist,
        n_iter=5, cv=tscv, scoring='f1_macro', n_jobs=-1, random_state=42
    )
    clf.fit(X, y)
    best = clf.best_estimator_
    logging.info(f'Trained model with params: {clf.best_params_}')
    joblib.dump(best, MODEL_PATH)
    return best

try:
    model = joblib.load(MODEL_PATH)
except (FileNotFoundError, EOFError):
    logging.warning('Model load failed (missing or corrupt). Retraining model.')
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    try:
        model = train_model()
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        logging.warning('Creating default RandomForestClassifier')
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        joblib.dump(model, MODEL_PATH)

# Helpers
def is_market_open():
    t = datetime.datetime.now().time()
    return datetime.time(9, 15) <= t <= datetime.time(15, 30)  # Regular NSE market hours: 9:15 AM - 3:30 PM IST

# Data cache
_data_cache = {'df': None, 'timestamp': None}
def get_latest_data():
    global _data_cache
    current_time = pd.Timestamp.now(tz='Asia/Kolkata').floor('1min')
    if _data_cache['df'] is not None and _data_cache['timestamp'] == current_time:
        logging.info("Using cached data")
        return _data_cache['df']
    max_retries = 3
    base_delay = 60
    for attempt in range(max_retries):
        try:
            df = yf.download('^NSEI', period='1d', interval='1m', auto_adjust=True)
            if df.empty:
                raise ValueError("No data returned from yfinance")
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tz_convert('Asia/Kolkata')
            df = df.sort_index()
            if not df.isna().any().any():
                df = calculate_indicators(df)
                _data_cache['df'] = df
                _data_cache['timestamp'] = current_time
                logging.info(f"Data fetched successfully from yfinance at {current_time}")
                return df
            logging.warning(f"Empty or invalid data on attempt {attempt + 1}")
        except Exception as e:
            logging.error(f"yfinance fetch error on attempt {attempt + 1}: {str(e)}")
            if "YFRateLimitError" in str(e):
                sleep_time = base_delay * (2 ** attempt)
                logging.info(f"Rate limit hit, retrying after {sleep_time} seconds")
                time.sleep(sleep_time)
            elif attempt == max_retries - 1:
                logging.error("Max retries reached. Returning None")
                return None
            else:
                time.sleep(base_delay)
    logging.error('Data fetch failed after all retries')
    return None

def get_vix():
    max_retries = 3
    base_delay = 60
    for attempt in range(max_retries):
        try:
            v = yf.download('^VIX', period='2d', interval='1d', auto_adjust=True)['Close']
            return v.iloc[-1]
        except Exception as e:
            logging.error(f"VIX fetch error on attempt {attempt + 1}: {e}")
            if "YFRateLimitError" in str(e):
                sleep_time = base_delay * (2 ** attempt)
                logging.info(f"Rate limit hit, retrying after {sleep_time} seconds")
                time.sleep(sleep_time)
            elif attempt == max_retries - 1:
                return None
            else:
                time.sleep(base_delay)
    return None

# Backtest CLI
def backtest_event(event_dt_str, interval='1m'):
    ts = pd.to_datetime(event_dt_str)
    start = ts - pd.Timedelta(minutes=3)
    end = ts + pd.Timedelta(days=1)
    if end.date() > datetime.datetime.now().date():
        end = datetime.datetime.now()
    max_retries = 3
    base_delay = 60
    for attempt in range(max_retries):
        try:
            df = yf.download('^NSEI', start=start, end=end, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError("No data returned from yfinance")
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tz_convert('Asia/Kolkata')
            df = df.sort_index()
            break
        except Exception as e:
            logging.error(f"Backtest data fetch error on attempt {attempt + 1}: {e}")
            if "YFRateLimitError" in str(e):
                sleep_time = base_delay * (2 ** attempt)
                logging.info(f"Rate limit hit, retrying after {sleep_time} seconds")
                time.sleep(sleep_time)
            elif attempt == max_retries - 1:
                print(f"No data for {ts} at interval={interval}")
                return
            else:
                time.sleep(base_delay)
    if df.empty:
        logging.error(f"No data for {ts} at interval={interval}; possibly a holiday")
        print(f"No data for {ts} at interval={interval}")
        return
    df = calculate_indicators(df)
    shift_bars = -3 if interval=='1m' else -1
    fut = df.shift(shift_bars)
    df['Breakout'] = np.where(fut['Close']>fut['Upper Band'],1,
                         np.where(fut['Close']<fut['Lower Band'],-1,0))
    df.dropna(inplace=True)
    feat_ts = start if start in df.index else df.index[df.index<=start][-1]
    feats = df.loc[feat_ts,FEATURES].values.reshape(1,-1)
    probs = model.predict_proba(feats)[0]
    cls = model.classes_
    idx = np.argmax(probs)
    pred, conf = cls[idx], probs[idx]
    true = df.loc[feat_ts,'Breakout']
    print(f"Event@{event_dt_str}: true={true}, pred={pred}, conf={conf:.2f}")
    print('✅' if pred==true and pred!=0 else '❌')

if __name__=='__main__':
    if 'backtest' in sys.argv:
        for arg in sys.argv[1:]:
            if arg.startswith('202'):
                backtest_event(arg)
        sys.exit(0)
    if not is_market_open() and datetime.datetime.now().weekday() >= 5:
        logging.info("Market closed on weekend. Sleeping for 1 hour.")
        time.sleep(3600)

# Main loop
last_log_time = datetime.datetime.now()
last_alert_time = None
while True:
    try:
        if not is_market_open():
            if datetime.datetime.now().weekday() >= 5:
                next_check = datetime.datetime.now() + datetime.timedelta(hours=1)
                logging.info(f"Market closed on weekend. Next check at {next_check.strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(3600)
            elif (datetime.datetime.now() - last_log_time).total_seconds() >= 3600:
                logging.info("Market closed. Waiting...")
                last_log_time = datetime.datetime.now()
            time.sleep(60)
            continue
        df = get_latest_data()
        if df is None or df.empty:
            logging.warning("Data fetch failed. Retrying after 60 seconds")
            time.sleep(60)
            continue
        last = df.iloc[-1]
        price, atr = last['Close'], last['ATR14']
        vix = get_vix()
        if vix is not None and vix > 20:
            logging.info(f"High VIX ({vix:.2f}); suppressing signals")
            time.sleep(60)
            continue
        elif vix is None:
            logging.info("VIX data unavailable, proceeding without VIX check.")
        # ML breakout alert
        x = df[FEATURES].iloc[-1].values.reshape(1,-1)
        probs = model.predict_proba(x)[0]
        pred = model.classes_[np.argmax(probs)]
        conf = max(probs)
        if pred != 0 and conf >= PROB_THRESH:
            if last_alert_time is None or (datetime.datetime.now() - last_alert_time).total_seconds() >= 300:
                tgt = price + 1.5 * atr if pred == 1 else price - 1.5 * atr
                send_message(f"ML_BREAKOUT_{'UP' if pred==1 else 'DOWN'} | Price:{price:.2f} | Target:{tgt:.2f} | ML_Conf:{conf:.2f}")
                logging.info('ML alert sent')
                last_alert_time = datetime.datetime.now()
        # Rule-based alerts
        if last['Close'] > last['Upper Band'] and last['VolSurge'] > 1.5 and last['RSI14'] > 50:
            tgt = price + PSY_MOVE_FACTOR * atr
            send_message(f"RULE_BREAKOUT_UP | Price:{price:.2f} | Target:{tgt:.2f} | Rule_Conf:{RULE_PROB_THRESH:.2f}")
            logging.info('Rule-based breakout UP alert sent')
        elif last['Close'] < last['Lower Band'] and last['VolSurge'] > 1.5 and last['RSI14'] < 50:
            tgt = price - PSY_MOVE_FACTOR * atr
            send_message(f"RULE_BREAKOUT_DOWN | Price:{price:.2f} | Target:{tgt:.2f} | Rule_Conf:{RULE_PROB_THRESH:.2f}")
            logging.info('Rule-based breakout DOWN alert sent')
        # 5-min ORB
        first_5min = df.between_time('09:15', '09:20')
        if not first_5min.empty:
            high_5min, low_5min = first_5min['High'].max(), first_5min['Low'].min()
            if price > high_5min and last['VolSurge'] > 1.5 and last['RSI14'] > 50 and last['MACD'] > last['MACD_Signal']:
                tgt = price + atr
                send_message(f"5MIN_ORB_UP | Price:{price:.2f} | Target:{tgt:.2f} | Rule_Conf:{RULE_PROB_THRESH:.2f}")
                logging.info('5-min ORB UP alert sent')
            elif price < low_5min and last['VolSurge'] > 1.5 and last['RSI14'] < 50 and last['MACD'] < last['MACD_Signal']:
                tgt = price - atr
                send_message(f"5MIN_ORB_DOWN | Price:{price:.2f} | Target:{tgt:.2f} | Rule_Conf:{RULE_PROB_THRESH:.2f}")
                logging.info('5-min ORB DOWN alert sent')
        # 15-min ORB
        first_15min = df.between_time('09:15', '09:30')
        if not first_15min.empty:
            high_15min, low_15min = first_15min['High'].max(), first_15min['Low'].min()
            if price > high_15min and last['VolSurge'] > 1.5 and last['RSI14'] > 50 and last['MACD'] > last['MACD_Signal']:
                tgt = price + FIFTEEN_MOVE_FACTOR * atr
                send_message(f"15MIN_ORB_UP | Price:{price:.2f} | Target:{tgt:.2f} | Rule_Conf:{RULE_PROB_THRESH:.2f}")
                logging.info('15-min ORB UP alert sent')
            elif price < low_15min and last['VolSurge'] > 1.5 and last['RSI14'] < 50 and last['MACD'] < last['MACD_Signal']:
                tgt = price - FIFTEEN_MOVE_FACTOR * atr
                send_message(f"15MIN_ORB_DOWN | Price:{price:.2f} | Target:{tgt:.2f} | Rule_Conf:{RULE_PROB_THRESH:.2f}")
                logging.info('15-min ORB DOWN alert sent')
        if (datetime.datetime.now() - last_log_time).total_seconds() >= 3600:
            logging.info('Trading system running')
            last_log_time = datetime.datetime.now()
        time.sleep(60)
    except Exception as e:
        logging.error(f"Loop error: {e}")
        time.sleep(60)

        
# import os
# import sys
# import logging
# from logging.handlers import RotatingFileHandler

# # Setup logging
# try:
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     LOG_PATH = os.path.join(BASE_DIR, 'trading_system.log')
#     handler = RotatingFileHandler(LOG_PATH, maxBytes=10*1024*1024, backupCount=5)
#     logging.basicConfig(handlers=[handler], level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#     logging.debug("Logging initialized")
# except Exception as e:
#     print(f"Logging setup failed: {e}")
#     sys.exit(1)

# try:
#     logging.debug("Script started")
#     logging.info("Minimal test running")
#     print("Minimal test running. Check ~/trading/trading_system.log")
#     logging.info("Minimal test completed")
#     sys.exit(0)
# except Exception as e:
#     logging.error(f"Startup error: {e}")
#     print(f"Startup error: {e}")
#     sys.exit(1)






# # main.py
# import os
# import sys
# import time
# import datetime
# import logging
# import joblib
# import asyncio
# import numpy as np
# import pandas as pd
# import yfinance as yf
# from dotenv import load_dotenv
# import telegram
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
# from indicators import calculate_indicators

# # Load environment
# load_dotenv()
# BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
# CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
# bot = telegram.Bot(token=BOT_TOKEN)

# # Sync wrapper for async send_message
# async def _async_send(text):
#     await bot.send_message(chat_id=CHAT_ID, text=text)

# def send_message(text):
#     loop = None
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         pass
#     if loop and loop.is_running():
#         asyncio.create_task(_async_send(text))
#     else:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         loop.run_until_complete(_async_send(text))

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'model.pkl'))
# LAST_INIT = os.path.join(BASE_DIR, 'last_init.txt')

# # Parameters
# PSY_MOVE_FACTOR = 1.0
# FIFTEEN_MOVE_FACTOR = 0.75
# PROB_THRESH = 0.75
# RULE_PROB_THRESH = 0.6
# FEATURES = [
#     'Close','SMA20','Upper Band','Lower Band','EMA20','EMA50',
#     'RSI14','MACD','MACD_Signal','MACD_Hist','ATR14','OBV','VWAP',
#     'VolSurge','Stoch%K','Stoch%D','CCI20','MFI14','CMF20','Price_to_Res'
# ]

# # Logging setup
# logging.basicConfig(filename=os.path.join(BASE_DIR, 'trading_system.log'),
#                     level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# today = datetime.datetime.now().strftime('%Y-%m-%d')
# if not os.path.exists(LAST_INIT) or open(LAST_INIT).read().strip() != today:
#     logging.info('Trading AI system initialized')
#     send_message('✅ Trading AI system initialized')
#     with open(LAST_INIT, 'w') as f:
#         f.write(today)

# # Model load/train

# def train_model():
#     # Fetch historical data for training
#     df = yf.download('^NSEI', period='14d', interval='1m', auto_adjust=True)
#     if df.empty:
#         logging.error('No historical data available for training.')
#         raise RuntimeError('Insufficient data to train model')
#     df = calculate_indicators(df)
#     # Label breakout up/down/none 3 minutes ahead
#     fut = df.shift(-3)
#     df['Breakout'] = np.where(
#         fut['Close'] > fut['Upper Band'], 1,
#         np.where(fut['Close'] < fut['Lower Band'], -1, 0)
#     )
#     df.dropna(inplace=True)

#     X, y = df[FEATURES], df['Breakout']
#     n_samples = len(df)
#     # Determine number of CV splits based on available samples
#     max_splits = min(5, n_samples // 10)  # at least 10 samples per split
#     if max_splits < 2:
#         logging.warning('Not enough samples for cross-validation; training without CV')
#         model = RandomForestClassifier(
#             n_estimators=100, random_state=42, class_weight='balanced'
#         )
#         model.fit(X, y)
#         joblib.dump(model, MODEL_PATH)
#         return model

#     # Hyperparameter tuning with time-series cross-validation
#     tscv = TimeSeriesSplit(n_splits=max_splits)
#     param_dist = {
#         'n_estimators': [100, 200],
#         'max_depth': [None, 10],
#         'min_samples_split': [2, 5]
#     }
#     clf = RandomizedSearchCV(
#         RandomForestClassifier(class_weight='balanced', random_state=42),
#         param_distributions=param_dist,
#         n_iter=5, cv=tscv, scoring='f1_macro', n_jobs=-1, random_state=42
#     )
#     clf.fit(X, y)
#     best = clf.best_estimator_
#     logging.info(f'Trained model with params: {clf.best_params_}')
#     joblib.dump(best, MODEL_PATH)
#     return best

# try:
#     model = joblib.load(MODEL_PATH)
# except (FileNotFoundError, EOFError):
#     logging.warning('Model load failed (missing or corrupt). Retraining model.')
#     # Remove corrupt model file if it exists
#     if os.path.exists(MODEL_PATH):
#         os.remove(MODEL_PATH)
#     model = train_model()

# # Helpers

# def is_market_open():
#     t = datetime.datetime.now().time()
#     return datetime.time(9,15) <= t <= datetime.time(15,30)

# def get_latest_data():
#     for i in range(3):
#         df = yf.download('^NSEI', period='1d', interval='1m', auto_adjust=True)
#         if not df.empty:
#             return calculate_indicators(df)
#         time.sleep(2**i)
#     logging.error('Data fetch failed')
#     return None

# def get_vix():
#     try:
#         v = yf.download('^VIX', period='2d', interval='1d', auto_adjust=True)['Close']
#         return v.iloc[-1]
#     except:
#         return None

# # Backtest CLI

# def backtest_event(event_dt_str, interval='1m'):
#     ts = pd.to_datetime(event_dt_str)
#     start = ts - pd.Timedelta(minutes=3)
#     hist = yf.download('^NSEI', start=(ts.date()-pd.Timedelta(days=1)).isoformat(),
#                         end=(ts.date()+pd.Timedelta(days=1)).isoformat(),
#                         interval=interval, auto_adjust=True)
#     if hist.empty:
#         print(f"No data for {ts} at interval={interval}")
#         return
#     hist = calculate_indicators(hist)
#     shift_bars = -3 if interval=='1m' else -1
#     fut = hist.shift(shift_bars)
#     hist['Breakout'] = np.where(fut['Close']>fut['Upper Band'],1,
#                          np.where(fut['Close']<fut['Lower Band'],-1,0))
#     hist.dropna(inplace=True)
#     feat_ts = start if start in hist.index else hist.index[hist.index<=start][-1]
#     feats = hist.loc[feat_ts,FEATURES].values.reshape(1,-1)
#     probs = model.predict_proba(feats)[0]
#     cls = model.classes_
#     idx = np.argmax(probs)
#     pred, conf = cls[idx], probs[idx]
#     true = hist.loc[feat_ts,'Breakout']
#     print(f"Event@{event_dt_str}: true={true}, pred={pred}, conf={conf:.2f}")
#     print('✅' if pred==true and pred!=0 else '❌')

# if __name__=='__main__':
#     if 'backtest' in sys.argv:
#         for arg in sys.argv[1:]:
#             if arg.startswith('202'):
#                 backtest_event(arg)
#         sys.exit(0)

# # Main loop
# while True:
#     try:
#         if not is_market_open(): time.sleep(60); continue
#         df = get_latest_data()
#         if df is None: time.sleep(60); continue
#         last = df.iloc[-1]
#         price, atr = last['Close'], last['ATR14']
#         # ML breakout alert
#         x = df[FEATURES].iloc[-1].values.reshape(1,-1)
#         probs = model.predict_proba(x)[0]
#         pred = model.classes_[np.argmax(probs)]; conf = max(probs)
#         if pred!=0 and conf>=PROB_THRESH:
#             tgt = price + 1.5*atr if pred==1 else price - 1.5*atr
#             send_message(f"ML_BREAKOUT_{'UP' if pred==1 else 'DOWN'} | Price:{price:.2f} | Target:{tgt:.2f} | ML_Conf:{conf:.2f}")
#             logging.info('ML alert sent')
#         # TODO: Add custom rule-based alerts back here
#         time.sleep(60)
#     except Exception as e:
#         logging.error(f"Loop error: {e}")
#         time.sleep(60)