## Env

The main goal of Gym is to provide a rich collection of environments for RL experiments using a unified interface, called **Env**.

At a high level, every environment provides these pieces of information and functionality:

* A set of **actions** that is allowed to be executed in the environment. Gym supports both discrete and continuous actions, as well as their combination.
* The shape and boundaries of the **observations** that the environment provides the agent with
* A method called **step** to execute an action, which returns the current observation, the reward, and a flag indicating that the episode is over.
* A method called **reset**, which returns the environment to its initial state and obtains the first observation.

## 动作空间和观察空间的抽象基类Space

![1770022967288](image/ch2_OpenaigymAPIandGymnasium/1770022967288.png)

## 什么是 Space 类？

**Space 类**是 Gymnasium 中用于**形式化定义环境输入/输出边界和结构的核心抽象**。

简单来说，它回答了三个关键问题：

1. **智能体可以执行哪些动作？** (动作空间 `action_space`)
2. **环境会返回什么样的观察？** (观察空间 `observation_space`)
3. 如何在这些空间内**合法地采样**和**验证**数据？

图中的 `Space` 类是所有具体空间类型的**抽象基类**，定义了一套统一的接口。

---

## **基类：`Space` 的四大核心成员**

如图，抽象的 `Space` 类提供了四个关键的属性/方法：

### **1. `shape: Tuple[int, ...]`**

- **含义**：空间的形状，与 NumPy 数组的 `shape` 概念完全一致。
- **作用**：明确数据的维度结构。
- **例子**：
  - `Box(3,)` → `shape = (3,)` （一维向量，长度3）
  - `Box(84, 84, 3)` → `shape = (84, 84, 3)` （RGB图像）
  - `Discrete(5)` → `shape = ()` （标量，0维）

### **2. `sample()`**

- **含义**：从该空间中**均匀随机采样**一个合法的元素。
- **作用**：
  - **测试环境**：快速获取一个合法的动作或观察示例。
  - **初始化/随机策略**：智能体在最开始时可以采用随机动作。
  - **探索**：作为探索策略的基础。
- **例子**：
  ```python
  action = env.action_space.sample() # 随机选择一个合法动作
  ```

### **3. `contains(x)`**

- **含义**：检查给定的输入 `x` **是否属于这个空间的定义域**（即是否合法）。
- **作用**：
  - **验证输入**：在智能体输出动作后，可以验证其是否在允许的范围内。
  - **调试**：确保自定义的环境或智能体没有产生非法数据。
- **例子**：
  ```python
  if not env.action_space.contains(my_action):
      raise ValueError(f"非法动作: {my_action}")
  ```

### **4. `seed(seed=None)`**

- **含义**：为空间内部的随机数生成器设置种子。
- **作用**：**保证可重复性**。设置相同的种子后，`sample()` 方法将产生完全相同的随机序列。
- **重要性**：在科学实验中，这是确保结果可复现的关键步骤。
- **例子**：
  ```python
  env.action_space.seed(42) # 固定动作空间的随机种子
  ```

---

## **三、主要子类详解**

图中展示了三个最常用的具体空间类型：

### **1. `Box` 空间**

- **用途**：表示**连续、多维的数值空间**。最常用。
- **关键属性**：
  - `low`: 一个标量或与 `shape` 同形的数组，定义每个维度的**下限**。
  - `high`: 定义每个维度的**上限**。
- **例子**：
  ```python
  # 1. 机械臂关节角度（3个关节，每个在 -π 到 π 之间）
  action_space = Box(low=-np.pi, high=np.pi, shape=(3,))

  # 2. Atari 游戏的图像观察（210x160 的 RGB 图像，像素值 0-255）
  observation_space = Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)

  # 3. 不同维度有不同范围（如机器人位置和速度）
  #    位置 x,y ∈ [-1, 1]，速度 vx,vy ∈ [-10, 10]
  obs_space = Box(low=np.array([-1.0, -1.0, -10.0, -10.0]),
                  high=np.array([1.0, 1.0, 10.0, 10.0]))
  ```

### **2. `Discrete` 空间**

- **用途**：表示**离散的、有限的选项集合**。通常用于分类动作。
- **关键属性**：
  - `n: int`：选项的数量。空间包含整数 `{0, 1, 2, ..., n-1}`。
- **例子**：
  ```python
  # 1. 简单的上下左右移动（4个动作）
  action_space = Discrete(4)
  # 动作 0: 上，1: 右，2: 下，3: 左

  # 2. 游戏中的按钮选择（如“开火”、“跳跃”、“蹲下”）
  action_space = Discrete(3)

  # 注意：Discrete 空间的样本是整数标量，如 `2`
  ```

### **3. `Tuple` 空间**

- **用途**：将**多个子空间组合**成一个复合空间。用于描述结构复杂的观察或复合动作。
- **关键属性**：
  - `spaces: Tuple[Space, ...]`：一个由子空间组成的元组。
- **例子**：
  ```python
  # 一个复杂的观察：包含图像、雷达数据和自身状态
  observation_space = Tuple((
      Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8), # 图像
      Box(low=0, high=100, shape=(10,)),                       # 10个雷达距离读数
      Discrete(4)                                              # 自身方向：北、东、南、西
  ))
  # 对应的观察是一个元组：(image_array, radar_array, direction_int)
  ```

---

## **四、其他重要 Space 子类（图中未展示但常用）**

### **`MultiBinary`**

- 表示 `n` 维的二进制向量（每个元素是 0 或 1）。
- **例子**：`MultiBinary(5)` 可以表示 5 个开关的开闭状态。

### **`MultiDiscrete`**

- 表示**多个独立的 `Discrete` 空间**。
- **例子**：`MultiDiscrete([5, 2, 3])` 表示三个离散选择，第一个有5个选项，第二个有2个，第三个有3个。常用于同时按多个按钮的游戏。

### **`Dict`**

- 与 `Tuple` 类似，但使用**字典键值对**来组织子空间。使数据访问更清晰。
- **例子**：
  ```python
  observation_space = Dict({
      “image”: Box(...),
      “lidar”: Box(...),
      “inventory”: Discrete(10)
  })
  ```

---

## **五、Space 类在 RL 工作流中的作用**

```python
import gymnasium as gym

env = gym.make(‘CartPole-v1’)

# 1. 查看空间定义（理解环境的第一步）
print(“动作空间:”, env.action_space)       # Discrete(2)
print(“观察空间:”, env.observation_space) # Box([-4.8, -∞, -0.42, -∞], [4.8, ∞, 0.42, ∞])， 形状 (4,)

# 2. 采样随机动作（用于初始化或探索）
random_action = env.action_space.sample()

# 3. 验证自定义动作是否合法
my_action = 2
if env.action_space.contains(my_action):
    print(“动作合法”)
else:
    print(f”动作 {my_action} 非法！允许范围是 {env.action_space}”)

# 4. 设置种子以复现实验
env.action_space.seed(123)
env.observation_space.seed(123)
```

#### **六、总结**

Gymnasium 的 **Space 类体系** 是连接**环境**与**智能体算法**的**强类型接口**，其核心价值在于：

1. **明确规范**：像 API 文档一样，精确定义了环境输入/输出的“数据类型”。
2. **自动验证**：通过 `contains()` 可以在运行时及早发现错误。
3. **便捷工具**：`sample()` 和 `seed()` 为开发和测试提供了极大便利。
4. **算法通用性**：RL 算法可以根据 `space` 的类型（如 `Box` 或 `Discrete`）自动选择合适的网络输出层（如高斯分布或 Softmax）。

**理解 Space 类是编写 Gymnasium 兼容环境和智能体的第一步，也是理解一个 RL 问题基本维度的关键。** 它把“智能体能做什么、能感知什么”这个抽象问题，转化为了清晰、可操作的数据结构定义。

## Env类

* action_space: This is the field of the Space class and provides a specification for allowed actions in the environment.
* observation_space: This field has the same Space class, but specifies the observations provided by the  environment.
* reset(): This resets the environment to its initial state, returning the initial observation vector and the dict with extra information from the environment.
* step(): This method allows the agent to take the action and returns information about the outcome of the action:

  – The next observation

  – The local reward

  – The end-of-episode flag

  – The flag indicating a truncated episode

  – A dictionary with extra information from the environment

运行step，算法内部执行过程：

* nender

## 装饰器-Wrappers

Very frequently, you will want to extend the environment’s functionality in some generic way.

There are many such situations that have the same

structure – you want to “wrap” the existing environment and add some extra logic for doing something. Gym

provides a convenient framework for this – the Wrapper class.

![1770024091352](image/ch2_OpenaigymAPIandGymnasium/1770024091352.png)




## ref

https://gymnasium.org.cn/introduction/basic_usage/
