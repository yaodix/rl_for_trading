import pandas as pd



# 读取 1 分钟 K 线数据
dir = "/home/yao/myproject/rl_for_trading/Rl_trading_demo/project_one/data/"
df_1min = pd.read_csv(dir + "159985.SZ.1m.csv") 


# 所有数据合成 5 分钟 K 线
df_5min = df_1min.resample("5min", on="trade_time").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "amount": "sum",
})


# 合成 15 分钟 K 线
df_15min = df_1min.resample("15min", on="trade_time").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "amount": "sum",
})

print(df_15min.tail(20))

#保存
# df_5min.to_csv("159985.SZ.5m.csv", index=False)
# df_15min.to_csv("159985.SZ.15m.csv", index=False)
