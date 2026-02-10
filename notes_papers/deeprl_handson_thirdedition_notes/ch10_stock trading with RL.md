### 强化学习环境设计

**The question is:** can we look at the problem from the RL angle? Let’s say that we have some observation of

the market, and we want to make a decision: buy, sell, or wait. If we buy before the price goes up, our profit

will be positive; otherwise, we will get a negative reward. What we’re trying to do is get as much profit as

possible. The connections between market trading and RL are quite obvious. First, let’s define the problem

statement more clearly

To formulate RL problems, three things are needed: observation of the environment,

possible actions, and a reward system

• Observation: The observation will include the following information:

– N past bars, where each has open, high, low, and close prices

– An indication that the share was bought some time ago (only one share at a time will be possible)

– The profit or loss that we currently have from our current position (the share bought)

• Action: At every step, after every minute’s bar, the agent can take one of the following actions:

– Do nothing: Skip the bar without taking an action

– Buy a share: If the agent has already got the share, nothing will be bought; otherwise, we will

pay the commission, which is usually some small percentage of the current price

– Close the position: if we do not have a previously purchased share, nothing will happen;

otherwise, we will pay the commission for the trade

• Reward: The reward that the agent receives can be expressed in various ways:

– As the first option, we can split the reward into multiple steps during our ownership of the share.

In that case, the reward on every step will be equal to the last bar’s movement.

– Alternatively, the agent can receive the reward only after the close action and get the full reward

at once.


### 训练改进思路
