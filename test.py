import numpy as np

data = np.array([
    [100, 1000, 50, 0.1, 99],
    [102, 1200, 55, 0.2, 100],
    [98,  800,  45, -0.1, 99],
    [101, 1100, 52, 0.15, 100]
])

def get_state(data, t, n):    
    d = t - n
    if d >= 0:
        block = data[d:t] 
    else:
        block =  np.array([data[0]]*n) 
    res = []
    for i in range(n - 1):
        feature_res = []
        for feature in range(data.shape[1]):
            feature_res.append((block[i + 1, feature] - block[i, feature]))
        res.append(feature_res)
    # display(res)
    return np.array([res])

# 获取t=3, n=3的状态
state = get_state(data, t=3, n=3)

print(state)