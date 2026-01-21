### src

肖老师的退休生活:

https://www.bilibili.com/video/BV1jK4y1w7T8?vd_source=bd54e8401db52a3c1b4d1b6be662966a&spm_id_from=333.788.videopod.sections

github: https://github.com/xccds/Ten_Minute_RL.git

一、引出冰湖问题，包括环境、状态转移等

二、为了解决这个问题设计一个随机策略

三、随机策略不好，人为设计了一个策略

四、人为设计的策略还是不够好，怎么寻找最优策略

五、想找最优策略，首先得知道一个方法来评估策略，比如value，Q值

六、知道如何评估后，需要有一种方法来改进策略

关于07_simple_dqn 网络的一个问题：

08_

# 十分钟强化学习第九讲：DQN的改进

### Double DQN

* Q-Learning是用max来估计Q值的，估计Q值偏高过于乐观。DQN也有同样的问题
* 一种解决思路是使用两个网络，一个用来选择max的Q值对应的action，另一个来负责输出对应的Q值，二者合作更为谨慎。
* DQN已经有了一个target Network，所以直接用它了。

### Dueling DQN

* 可以将Q进行拆解： Q = V + A，将状态和行动的收益分开来建模。
* 分别用神经网络来近似，更好的学习到Q

### Prioritized replay buffer

* 过去的经验中有重要的有不重要
* 对重要的经验进行加权抽样，反复进行学习
* 用abs(error)当做经验的重要性度量
