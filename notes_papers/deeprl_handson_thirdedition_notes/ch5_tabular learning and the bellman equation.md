focus：familiar with the Bellman equation and the practical method of its application

In this chapter, we will:

• Review the value of the state and the value of the action, and learn how to calculate them in simple cases

• Talk about the Bellman equation and how it establishes the optimal policy if we know the values of states

• Discuss the value iteration method and try it on the FrozenLake environment

• Do the same for the Q-iteration method

called value iteration.

V值更新过程：

![1770089312362](image/ch5_tabularlearningandthebellmanequation/1770089312362.png)

由公式可知，我们更新过程中需要移植奖励r 和 转移概率p。

在实际的**程序实现**中，我们通常会按状态编号顺序计算（从0到n），但：

1. **每个状态的计算公式依赖于后续状态的值**
2. 我们使用的是 **上一轮迭代的旧值** ，所以可以并行计算所有状态
3. **价值传播在数学上是逆向的** ，但编码计算可以是任何顺序

在值函数的数学框架下，非零奖励是系统中唯一的“价值源泉”。如果这个源泉从未被触发（即从未到达目标），那么整个状态空间的价值评估就全是0，学习完全停滞。

以下也表示最优价值，略有不同，来自华泰59.

![1770099519011](image/ch5_tabularlearningandthebellmanequation/1770099519011.png)Q值相对于V值计算的核心好处在于：Q值直接编码了动作的价值信息，使决策更直观，更适合无模型强化学习场景。

Q值迭代过程参考上述q(s, a)方程，同样需要状态转移概率矩阵P。
