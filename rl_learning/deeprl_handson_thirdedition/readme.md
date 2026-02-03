gymnasuim 1.2.3

pip install gymnasium[atari]

RL改进过程：

ch5中value_iteration和q_iteration 迭代改进所有的状态或状态-动作(full set of states). 代码实现为state和action的2层for循环。

q_learing一定程度上解决了遍历所有状态的情况，但是当状态很多的时候任然很挣扎，代码上使用dict{key(state, action) : value }表示待更新的状态，更新过程采用迭代的方式。

引出DQN：

DQN扩展：未来让DQN更好更稳定的训练


安装
