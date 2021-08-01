import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

window_size = 120

base = 345600 * 10

m1_args = {"symbol": None, "timeframe": mt5.TIMEFRAME_M1, "start_pos": 0, "count": base}
m5_args = {"symbol": None, "timeframe": mt5.TIMEFRAME_M5, "start_pos": 0, "count": base // 5}
m15_args = {"symbol": None, "timeframe": mt5.TIMEFRAME_M15, "start_pos": 0, "count": base // 15}
m30_args = {"symbol": None, "timeframe": mt5.TIMEFRAME_M30, "start_pos": 0, "count": bse // 30}
h1_args = {"symbol": None, "timeframe": mt5.TIMEFRAME_H1, "start_pos": 0, "count": base // 60}
h4_args = {"symbol": None, "timeframe": mt5.TIMEFRAME_H4, "start_pos": 0, "count": base // 240}
d1_args = {"symbol": None, "timeframe": mt5.TIMEFRAME_D1, "start_pos": 0, "count": base // 1440}


def gen_data(symbol, time_args=m1_args, pred_length=1):
    """
    symbol = "EURUSD", "USDJPY"...
    window_size = 30, 60, 120, ...
    time_args= m1_args, m5_args, ...
    """
    init = mt5.initialize()
    assert init == True
    
    time_args.update({"symbol": symbol})
    r = mt5.copy_rates_from_pos(**time_args)
    df = pd.DataFrame(r)

    df["close"] = MinMaxScaler().fit_transform(df["close"])
    lists = ["close"]
    x = df[lists]
    x = np.array(x)

    y = np.array(df["close"])
    
    time_x = []
    time_y = []

    for i in range(len(y) - pred_length + 1):
        if i > window_size:
            time_x.append(x[i - window_size:i])
            time_y.append(y[i:i+pred_length])

    x, y = np.array(time_x).reshape((-1, window_size, 1)), np.array(time_y).reshape((-1, pred_length))
    
    return train_test_split(x, y, test_size=0.33, random_state=42)
