import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

symbol=["AUDUSD", "AUDJPY", "NZDJPY", "GBPJPY", "EURUSD", "GBPUSD", "NZDUSD", "USDJPY", "EURJPY"]


"""
install Metatrader5 and create demo account(example XMTrading)

pip install MetaTrader5
"""

def gen_data(symbol=symbol):
    init = mt5.initialize()
    assert init == True
    x_list = []
    y_list = []
    atr = []

    for s in symbol:

        while True:
            try:
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
         
        df["close"] = MinMaxScaler().fit_transform(df["close"])
        lists = ["close"]
        shape = len(lists)
        x = df[lists]
        x = np.array(x)
        print(x.shape)

        y = np.array(df["close"])

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
