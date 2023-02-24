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
        print(state,action)
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
        return  self.state 

class TemporalDifference:
    def __init__(self,env,alpha,gamma,epsilon) -> None:
        self.env = env  # environment 
        self.gamma = gamma #discount factor 
        self.alpha = alpha  #step size 
        self.epsilon = epsilon
        pass
    def initialize(self):
        Q = {}
        for state in self.env.states:
            # Q[state] =
            # print(self.env.states)
            # print(self.env.environment[state])
            state = tuple(state)
            # if self.env.environment[state]!=1: #checking is that action can be taken because of the boundaries 
            Q[state] = {action:0 for action in self.env.actions if self.env.tuple_sum(state,action) in map(tuple,self.env.states) }
            # else: # if it is a cliff , then there is no action , just bounce back and reset 
            #     Q[state] = {} 
        return Q 
    
    def epsilon_greedy(self,Q,S):
        if np.random.random()<=self.epsilon: 
                # Explore 
                A = np.random.choice(list(Q[S]))
        else:
            #Exploit
            A = max(Q[S], key=Q[S].get, default=None)
        return A 
    
    def SARSA(self,n_episodes):

        Q = self.initialize() # initialize Q(s,a) for all s and a except for the terminal states 
        # print(Q)
        for i in range(n_episodes): # loop for each episode
            S = self.env.reset() # Initialize state. this is basically start state 
            # print(S)
            # loop for each step of the episode until S is terminal 
            while True:
                # choose A from S using epsilon-greedy 
                A = self.epsilon_greedy(Q,S)
                # print(S,A)
                # take action A , observe R and S' 
                S_ = self.env.next_state(S,A)
                R = self.env.reward(S_)
                # Choose A' from S' using policy derived from Q - epsilon greedy 
                A_ = self.epsilon_greedy(Q,S_)
                Q[S][A] = Q[S][A] + self.alpha*(R+self.gamma*Q[S_][A_]-Q[S][A])
                S = S_
                A = A_ 

                if self.env.is_terminal(S):
                    break 
        return Q 
    
    def QLearning(self,n_episodes):
        # only one step changes compared to SARSA 
        Q = self.initialize() # initialize Q(s,a) for all s and a except for the terminal states 

        for i in range(n_episodes): # loop for each episode
            S = self.env.reset() # Initialize state. this is basically start state 
            # print(S)
            # loop for each step of the episode until S is terminal 
            while True:
                # choose A from S using epsilon-greedy 
                A = self.epsilon_greedy(Q,S)
                # print(S,A)
                # take action A , observe R and S' 
                S_ = self.env.next_state(S,A)
                R = self.env.reward(S_)

                # print(max(Q[S_]))
                
                Q[S][A] = Q[S][A] + self.alpha*(R+self.gamma*max(Q[S_].values())-Q[S][A])
                S = S_
                if self.env.is_terminal(S):
                    break 
        return Q

def main():
    # actions = [""]
    environment = np.loadtxt("cliff.txt")
    start = (3,0)
    goal = (3,11)
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1 
    n_episodes = 1000

    env = Env(environment,start,goal)

    TD = TemporalDifference(env,alpha,gamma,epsilon)
    # Q = TD.SARSA(n_episodes)
    Q = TD.QLearning(n_episodes)
    print(Q)
    # for state in env.states:
    #     print(state)

    #     print(env.is_terminal(tuple(state)))
    # print(env.next_state(goal,"u"))

    return 

if __name__ == "__main__":
    main()