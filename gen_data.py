import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import ta
from ta.trend import _ema as ema
from ta.momentum import stoch, stoch_signal
# from ta.trend import
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore

symbol=["AUDUSD", "AUDJPY", "NZDJPY", "GBPJPY"] + ["EURUSD", "GBPUSD", "NZDUSD", "USDJPY", "EURJPY"]


"""
install Metatrader5 and create demo account(example XMTrading)

pip install MetaTrader5
"""


def fast_stochastic(lowp, highp, closep, period=14, smoothing=3):
    """ calculate slow stochastic
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K
    """
    low_min = lowp.rolling(window=period).min()
    high_max = highp.rolling(window=period).max()
    k_fast = 100 * (closep - low_min)/(high_max - low_min)
    d_fast = k_fast.rolling(window = smoothing).mean()
    d_slow = d_fast.rolling(window=smoothing).mean()
    return d_fast, d_slow


def gen_data(symbol=symbol):
    init = mt5.initialize()
    assert init == True
    x_list = []
    y_list = []
    atr = []

    for s in symbol:

        while True:
            try:
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M30, 0, 11520 * 10)
                r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M15, 0, 23040 * 10)
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, 69120 * 5)
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_H1, 0, 5760 * 10)
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_H4, 0, 1440 * 5)
                # r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_D1, 0, 240 * 3)
                df = pd.DataFrame(r)
                df.close
                break
            except:
                pass
                # print(1)
        try:
            print(s)
            point = mt5.symbol_info(s).point
            str_point = str(point)
            time = df["time"]
            if "e-" in str_point:
                point = int(str_point[-1])
                print(point)
                df *= 10 ** point
            else:
                point = len(str_point.rsplit(".")[1])
                if point == 1:
                    point = 0
                df *= 10 ** point
            # df = np.round(df, 0)
            df["time"] = pd.to_datetime(time, unit='s')
            df.index = df.time
            df.index.name = "Date"
            df = df[["open", "high", "low", "close", "tick_volume"]]
        except:
            pass

        df["sig"] = df.close - df.close.shift(1)
        df["sig2"] = df.close - df.close.shift(5)
        df["sig3"] = df.close - df.close.shift(10)
        df["ema"] = ta.trend._ema(df.close, 12) - ta.trend._ema(df.close, 26)
        df["ema2"] = ta.trend._ema(df.close, 26) - ta.trend._ema(df.close, 51)
        df["ema3"] = ta.trend._ema(df.close, 51) - ta.trend._ema(df.close, 100)
        df["macd"] = df.ema - ta.trend._ema(df.ema, 9)
        df["macd2"] = df.ema2 - ta.trend._ema(df.ema2, 6)
        df["eom"] = ta.volume.ease_of_movement(df.high, df.low, df.tick_volume)
        df["rsi"] = ta.momentum.rsi(df.close)# - 50
        df["sig_rsi"] = df.rsi - df.rsi.shift(1)
        df["atr"] = ta.volatility.average_true_range(df.high, df.low, df.close)
        base = ta.trend.ichimoku_base_line(df.high, df.low)
        conversion = ta.trend.ichimoku_conversion_line(df.high, df.low)
        fs, ss = fast_stochastic(df.low, df.high, df.close)
        df["stoch"] = fs - ss
        df["ichimoku"] = conversion - base
        
        df["y"] = np.where(df.sig > 0, 0, 1)

        df["rsi"] -= 50

        df = df.dropna()

        lists = ["sig", "sig2", "sig3", "stoch"]
        shape = len(lists)
        x = df[lists]
        x = np.array(x)
        print(x.shape)

        y = np.array(df["y"])

        print("gen time series data")

        window_size = 120
        time_x = []
        time_y = []

        for i in range(len(y)):
            if i > window_size:
                time_x.append(x[i - window_size : i])
                time_y.append(y[i])
        
        x = np.array(time_x).reshape((-1, window_size, x.shape[-1]))
        y = np.array(time_y).reshape((-1,))

        x_list.append(x)
        y_list.append(y)
    
    np.save("x", np.array(x_list).astype(np.float32).reshape(-1, window_size, shape))
    np.save("target", np.array(y_list).astype(np.float32).reshape(-1,))

    print("done\n")


if __name__ == "__main__":
    gen_data()
