import pandas as pd


# 读取 30 分钟 K 线数据
input_file = "Rl_trading_demo/project_one/data/518880.SH.30m.csv"
df_30min = pd.read_csv(input_file)


print(df_30min.tail(20))
print(df_30min.head(20))
