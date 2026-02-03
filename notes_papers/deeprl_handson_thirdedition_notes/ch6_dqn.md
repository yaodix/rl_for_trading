in this chapter, we will:

• Talk about problems with the value iteration method and consider its variation, called Q-learning.

• Apply Q-learning to so-called grid world environments, which is called tabular Q-learning.

• Discuss Q-learning in conjunction with neural networks (NNs). This combination has the name deep Q-network (DQN)

### problems with the value iteration method：

* 不适用state巨大的环境
* 不适用连续动作空间

If some state in the state space is not shown to us by the environment, why should we care about its

value? We can only use states obtained from the environment to update the values of states, which can save

us a lot of work。

This modification of the value iteration method is known as Q-learning,

注意这里迭代公式里已经不含有状态转移矩阵和求和了，无模型方法特征：

* 只使用**实际观察到的**下一状态 **s**′ 和奖励 **r**
* **不需要**转移概率 **P**
* **不需要**对所有可能状态求和

### **为什么可以省略求和？**

**核心原理：用样本估计期望**

无模型强化学习中，我们**不求和**是因为：

1. **用样本近似期望** ：实际观察到的 (s,a,r,s′)**(**s**,**a**,**r**,**s**′**) 是环境分布的一个样本
2. **随机梯度下降** ：我们使用这个样本进行更新，相当于用**蒙特卡洛方法估计期望**
3. **在期望意义上等价** ：如果采样足够多，样本平均会收敛到真实期望

![1770111144039](image/ch6_dqn/1770111144039.png)

### **Q-learning的巧妙绕过** ：

1. 不显式估计 **P**(**s**′**∣**s**,**a**)**
2. 直接从经验 **(**s**,**a**,**r**,**s**′**) 学习
3. 用采样代替求和，用实际观察代替概率分布

### 使用DQN

1. 解决探索和利用问题
   * epsilon-greedy method
2. 解决训练稳定问题
   * target-net
3. 解决数据分布问题
   * replay buffer

![1770116912624](image/ch6_dqn/1770116912624.png)


DQN通过重新设计网络架构，从"输入状态-动作对，输出单个Q值"变为"只输入状态，一次性输出所有动作的Q值"，获得了显著的性能提升，这是深度Q学习能够成功应用于复杂环境的重要因素之一。
