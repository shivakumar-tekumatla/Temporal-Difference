"""
This gridworld example compares Sarsa and Q-learning, highlighting the difference between on-policy (Sarsa) and off-policy (Q-learning) methods.
Consider the gridworld shown to the right. This is a standard undiscounted, episodic task, with start and goal states, and the usual actions causing movement up, down,
right, and left. Reward is -1 on all transitions except those into the region marked “The Cliff.” 
Stepping into this region incurs a reward of -100 and sends the agent instantly back to the start.

"""
import numpy as np 

from itertools import product
class Env:
    def __init__(self,environment,start,goal) -> None:
        self.environment = environment # 
        self.x_limit,self.y_limit = self.environment.shape
        self.states = np.array([state for state in product(np.arange(self.x_limit),np.arange(self.y_limit ))]) # every index of the environment! 
        self.actions = {"r":(0,1),
                        "l":(0,-1),
                        "d":(1,0),
                        "u":(-1,0)}  # basically there are four actions ,and these are represented as keys and the values re nothing but the indices by which state will change 
        self.start= start#(3,0)  
        self.state = self.start # by default the state is start, and it will be updated over the time 
        self.goal = goal #(3,11) 
        self.tuple_sum = lambda state,action:tuple(map(sum,zip(state,self.actions[action]))) #does tuple sum 
        # self.next_state = self.start
        pass
    
    def is_terminal(self,state):
        # state is in the form of indices 

        if (self.environment[state]) == 1 or (state == self.goal):
            return True 
        return False 

    def next_state(self,state,action):
        return self.tuple_sum(state,action)

    def reward(self,next_state):
        if next_state == self.goal:
            return -1 
        elif self.is_terminal(next_state): # check if it is goal first, then terminal with -100 reward 
            return -100
        else:
            return -1 

    def reset(self):
        #Whenever the robot transitioned to terminal state , reset the position to start position 
        self.state = self.start
        return  self.self.state 

class TD:
    def __init__(self,env,alpha,gamma) -> None:
        self.env = env 
        self.gamma = gamma 
        self.alpha = alpha 
        pass
    def initialize(self):
        return

def main():
    # actions = [""]
    environment = np.loadtxt("cliff.txt")
    start = (3,0)
    goal = (3,11)

    env = Env(environment,start,goal)

    print(env.next_state(goal,"u"))

    return 

if __name__ == "__main__":
    main()