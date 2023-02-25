"""
This gridworld example compares Sarsa and Q-learning, highlighting the difference between on-policy (Sarsa) and off-policy (Q-learning) methods.
Consider the gridworld shown to the right. This is a standard undiscounted, episodic task, with start and goal states, and the usual actions causing movement up, down,
right, and left. Reward is -1 on all transitions except those into the region marked “The Cliff.” 
Stepping into this region incurs a reward of -100 and sends the agent instantly back to the start.

"""
import numpy as np 

from itertools import product
import matplotlib.pyplot as plt
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
        # print(state,action)
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
    def __init__(self,env,alpha,gamma,epsilon,eps_decay_factor =0.01) -> None:
        self.env = env  # environment 
        self.gamma = gamma #discount factor 
        self.alpha = alpha  #step size 
        self.epsilon = epsilon
        self.eps_decay_factor = eps_decay_factor # decay the epsilon after each episode by some factor 
        pass
    def initialize(self):
        Q = {}
        for state in self.env.states:
            state = tuple(state)
            #checking is that action can be taken because of the boundaries 
            Q[state] = {action:0 for action in self.env.actions if self.env.tuple_sum(state,action) in map(tuple,self.env.states) }
        return Q 
    
    def epsilon_greedy(self,Q,S):
        if np.random.random()<=self.epsilon: 
                # Explore 
                A = np.random.choice(list(Q[S]))
        else:
            #Exploit
            A = max(Q[S], key=Q[S].get, default=None)
        return A 
    
    def SARSA(self,n_episodes,eps_greedy_decay=True):

        Q = self.initialize() # initialize Q(s,a) for all s and a except for the terminal states 
        rewards = []
        for i in range(n_episodes): # loop for each episode
            S = self.env.reset() # Initialize state. this is basically start state 
            # loop for each step of the episode until S is terminal 
            total_R = 0
            while True:
                # choose A from S using epsilon-greedy 
                A = self.epsilon_greedy(Q,S)
                # print(S,A)
                # take action A , observe R and S' 
                S_ = self.env.next_state(S,A)
                R = self.env.reward(S_)
                total_R +=R
                # Choose A' from S' using policy derived from Q - epsilon greedy 
                A_ = self.epsilon_greedy(Q,S_)
                Q[S][A] = Q[S][A] + self.alpha*(R+self.gamma*Q[S_][A_]-Q[S][A])
                S = S_
                A = A_ 

                if self.env.is_terminal(S):
                    break 
            if eps_greedy_decay:
                # decay the epsilon after each episode 
                self.epsilon = self.epsilon*self.eps_decay_factor
            rewards.append(total_R)
        return Q,rewards
    
    def QLearning(self,n_episodes,eps_greedy_decay=True):
        # only one step changes compared to SARSA 
        Q = self.initialize() # initialize Q(s,a) for all s and a except for the terminal states 
        rewards = []
        for i in range(n_episodes): # loop for each episode
            S = self.env.reset() # Initialize state. this is basically start state 
            total_R = 0
            # print(S)
            # loop for each step of the episode until S is terminal 
            while True:
                # choose A from S using epsilon-greedy 
                A = self.epsilon_greedy(Q,S)
                # print(S,A)
                # take action A , observe R and S' 
                S_ = self.env.next_state(S,A)
                R = self.env.reward(S_)
                total_R +=R
                # print(max(Q[S_]))
                
                Q[S][A] = Q[S][A] + self.alpha*(R+self.gamma*max(Q[S_].values())-Q[S][A])
                S = S_
                if self.env.is_terminal(S):
                    break 
            if eps_greedy_decay:
                # decay the epsilon after each episode 
                self.epsilon = self.epsilon*self.eps_decay_factor
            rewards.append(total_R)
        return Q,rewards
    def plot_path(self,Q,start,goal,title):
        plt.title(title)
        path = self.trace(Q,start,goal)
        plt.imshow(1-self.env.environment,cmap="gray")
        gy,gx = goal 
        sy,sx = start
        plt.plot(sx,sy,"ro")
        plt.plot(gx,gy,"go")
        for i in range(0,len(path)-1,2):
            S,A = path[i],path[i+1]
            y,x = S
            dy,dx = self.env.actions[A]
            mul = 0.5
            plt.arrow(x,y,mul*dx,mul*dy,width=0.05)

        plt.show()
        return None
    def trace(self,Q,start,goal):
        # tracing the actions with max value 
        path = [] 
        S = start 
        while S != goal: 
            path.append(S)
            A = max(Q[S], key=Q[S].get, default=None)
            path.append(A)
            S = self.env.next_state(S,A)
        return path 


def main():
    # actions = [""]
    environment = np.loadtxt("cliff.txt")
    start = (3,0)
    goal = (3,11)
    alpha = 0.2
    gamma = 0.9
    epsilon = 0.1
    n_episodes = 1000
    episodes = [i for i in range(n_episodes)]

    env = Env(environment,start,goal)

    TD = TemporalDifference(env,alpha,gamma,epsilon)
    Q,rewards1 = TD.SARSA(n_episodes,eps_greedy_decay=False)
    TD.plot_path(Q,start,goal,"SARSA Without Epsilon Decay")

    Q,rewards2 = TD.QLearning(n_episodes,eps_greedy_decay=False)
    TD.plot_path(Q,start,goal,"Q Learning Without Epsilon Decay")
    # plt.plot(episodes,rewards1)
    # plt.legend(["SARSA","Q Learning"])
    # plt.show()
    Q,rewards3= TD.SARSA(n_episodes)
    TD.plot_path(Q,start,goal,"SARSA With Epsilon Decay")
    Q,rewards4 = TD.QLearning(n_episodes)
    TD.plot_path(Q,start,goal,"Q Learning With Epsilon Decay")
    plt.plot(episodes,rewards1)
    plt.plot(episodes,rewards2)
    plt.plot(episodes,rewards3)
    plt.plot(episodes,rewards4)
    plt.legend(["SARSA","Q Learning","S2","Q2"])
    plt.show()
    
    return 

if __name__ == "__main__":
    main()