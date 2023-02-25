# Temporal-Difference
Cliff Walking Problem using SARSA and Q-Learning!

This gridworld example compares Sarsa and Q-learning, highlighting the difference between on-policy (Sarsa) and off-policy (Q-learning) methods. Consider the gridworld below. This is a standard undiscounted, episodic task, with start and goal states, and the usual actions causing movement up, down, right, and left. Reward is -1 on all transitions except those into the region marked “The Cliff.”  Stepping into this region incurs a reward of -100 and sends the agent instantly back to the start. 

<img src="https://github.com/shivakumar-tekumatla/Temporal-Difference/blob/main/Outputs/world.png" width="500">

The following hyper prameters are used:

```
    start = (3,0)
    goal = (3,11)
    alpha = 0.2
    gamma = 0.9
    epsilon = 0.1
    epsilon decay factor = 0.01
    n_episodes = 1000
```


When there is no decay of epsilon in epsilon-greedy , SARSA converges to a longer safer path , where as the Q-learning converges to an optimal path. This is given by the following images. 

<img src="https://github.com/shivakumar-tekumatla/Temporal-Difference/blob/main/Outputs/SARSA_no_decay.png" width="500">
<img src="https://github.com/shivakumar-tekumatla/Temporal-Difference/blob/main/Outputs/QL_no_decay.png" width="500">

But if the epsilon is gradually reduced after each episode, even the SARSA converges to the optimal path. 

<img src="https://github.com/shivakumar-tekumatla/Temporal-Difference/blob/main/Outputs/SARSA_decay.png" width="500">
<img src="https://github.com/shivakumar-tekumatla/Temporal-Difference/blob/main/Outputs/QL_decay.png" width="500">