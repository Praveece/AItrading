# # monitor.py
# import time
# import datetime
# import logging
# from logging.handlers import RotatingFileHandler
# import os
# from main import get_latest_data, model, FEATURES, PROB_THRESH, RULE_PROB_THRESH, send_message, is_market_open, get_vix

# # Logging setup
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# handler = RotatingFileHandler(os.path.join(BASE_DIR, 'trading_system.log'), maxBytes=10*1024*1024, backupCount=5)
# logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def classify(df):
#     X = df[FEATURES].iloc[-1:]
#     pred = model.predict(X)[0]
#     proba = model.predict_proba(X)[0]
#     confidence = max(proba)
#     return pred, confidence

# def rule_based(df):
#     row = df.iloc[-1]
#     volume_high = row['VolSurge'] > 1.5
#     # 5-min ORB
#     first_5min = df.between_time('09:15', '09:20')
#     if not first_5min.empty:
#         high_5min, low_5min = first_5min['High'].max(), first_5min['Low'].min()
#         if row['Close'] > high_5min and volume_high and row['RSI14'] > 55 and row['MACD_Hist'] > 0:
#             return 1
#         elif row['Close'] < low_5min and volume_high and row['RSI14'] < 45 and row['MACD_Hist'] < 0:
#             return -1
#     # General momentum
#     if volume_high and row['MACD_Hist'] > 0 and row['RSI14'] > 55 and row['Price_to_Res'] > 0.01:
#         return 1
#     elif volume_high and row['MACD_Hist'] < 0 and row['RSI14'] < 45:
#         return -1
#     return 0

# def monitor():
#     last_alert_time = None
#     while True:
#         try:
#             if not is_market_open():
#                 time.sleep(60)
#                 continue
#             vix = get_vix()
#             if vix is not None and vix > 20:
#                 logging.info(f"High VIX ({vix:.2f}); suppressing signals")
#                 time.sleep(60)
#                 continue
#             df = get_latest_data()
#             if df is None or len(df) < 3:
#                 time.sleep(60)
#                 continue
#             row = df.iloc[-1]
#             pred, confidence = classify(df)
#             rule_suggest = rule_based(df)
#             message = f"\n📈 Time: {datetime.datetime.now().strftime('%H:%M:%S')}\n"
#             if confidence >= PROB_THRESH:
#                 tgt = row['Close'] + 1.5 * row['ATR14'] if pred == 1 else row['Close'] - 1.5 * row['ATR14']
#                 sl = row['Close'] - row['ATR14'] if pred == 1 else row['Close'] + row['ATR14']
#                 message += f"🤖 Model Signal: {'BUY' if pred == 1 else 'SELL' if pred == -1 else 'HOLD'} (Conf: {confidence:.2f}, Target: {tgt:.2f}, Stop-Loss: {sl:.2f})\n"
#             if rule_suggest != 0:
#                 tgt = row['Close'] + row['ATR14'] if rule_suggest == 1 else row['Close'] - row['ATR14']
#                 sl = row['Close'] - row['ATR14'] if rule_suggest == 1 else row['Close'] + row['ATR14']
#                 message += f"📚 Rule-Based Signal: {'BUY' if rule_suggest == 1 else 'SELL'} (Target: {tgt:.2f}, Stop-Loss: {sl:.2f})\n"
#             if message.strip() != f"\n📈 Time: {datetime.datetime.now().strftime('%H:%M:%S')}\n":
#                 if last_alert_time is None or (datetime.datetime.now() - last_alert_time).total_seconds() >= 300:
#                     send_message(message)
#                     logging.info(f"Sent message: {message}")
#                     last_alert_time = datetime.datetime.now()
#             time.sleep(60)
#         except Exception as e:
#             logging.error(f"Monitor loop error: {e}")
#             time.sleep(60)

# if __name__ == '__main__':
#     monitor()



# # monitor.py (NEW)
# import time
# import datetime
# from main import get_latest_data, model, FEATURES, PROB_THRESH, RULE_PROB_THRESH, send_message

# def classify(df):
#     X = df[FEATURES].iloc[-1:]
#     pred = model.predict(X)[0]
#     proba = model.predict_proba(X)[0]
#     confidence = max(proba)
#     return pred, confidence

# def rule_based(df):
#     row = df.iloc[-1]
#     if row['MACD_Hist'] > 0 and row['RSI14'] > 55 and row['Price_to_Res'] > 0.01:
#         return 1
#     if row['MACD_Hist'] < 0 and row['RSI14'] < 45:
#         return -1
#     return 0

# def monitor():
#     while True:
#         if not is_market_open():
#             time.sleep(60)
#             continue

#         df = get_latest_data()
#         if df is None or len(df) < 3:
#             time.sleep(60)
#             continue

#         pred, confidence = classify(df)
#         rule_suggest = rule_based(df)

#         message = f"\n📈 Time: {datetime.datetime.now().strftime('%H:%M:%S')}\n"

#         if confidence >= PROB_THRESH:
#             message += f"🤖 Model Signal: {'BUY' if pred == 1 else 'SELL' if pred == -1 else 'HOLD'} (Conf: {confidence:.2f})\n"
#         if rule_suggest != 0:
#             message += f"📚 Rule-Based Signal: {'BUY' if rule_suggest == 1 else 'SELL'}\n"

#         if message.strip():
#             send_message(message)

#         time.sleep(60)  # Every minute

# if __name__ == '__main__':
#     monitor()