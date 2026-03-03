import pandas as pd

# 创建一个简单的DataFrame用于测试
df_test = pd.DataFrame({
    'Close': [100, 101, 102, 103, 104],
    'BB_Upper': [105, 106, 107, 108, 109],
    'BB_Lower': [95, 96, 97, 98, 99]
})

print(df_test)

print(df_test.shift(1))

print(0.5/10000)