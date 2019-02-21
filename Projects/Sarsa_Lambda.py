import random
import numpy as np

class Sarsa_Lambda():
    def __init__(self,state_space,action_space,LAMBDA,GAMMA,EPSILON_MAX=0.99,EPSILON_STEP=0.01,method="Tabular"):
        self.state_space = state_space
        self.action_space = action_space
        self.ALPHA = 0.5
        self.LAMBDA = LAMBDA
        self.GAMMA = GAMMA
        self.EPSILON = 0.50
        self.EPSILON_MAX = EPSILON_MAX
        self.EPSILON_STEP = EPSILON_STEP
        self.method = method

        # Action-Value Function
        self.Q = [[0 for a in range(action_space)] for s in range(state_space)]

        # Eligability Trace
        self.E = [[0 for a in range(action_space)] for s in range(state_space)]

    def act(self,state):
        action = None
        rand = random.randint(1,101)
        if(rand < (self.EPSILON * 100)):
            action = np.argmax(self.Q[state])
        else:
            action = random.randint(0,self.action_space-1)
        return action

    def epsilion_step(self):
        if(self.EPSILON != self.EPSILON_MAX):
            self.EPSILON = self.EPSILON + self.EPSILON_STEP
            if(self.EPSILON > self.EPSILON_MAX):
                self.EPSILON = self.EPSILON_MAX 

    def calc_delta(self,s,a,r,_s,_a):
        return r + self.GAMMA * self.Q[_s][_a] - self.Q[s][a]

    def increment_E(self,s,a):
        self.E[s][a] = self.E[s][a] + 1

    def update(self,s,a,r,_s,_a):
        delta = self.calc_delta(s,a,r,_s,_a)
        self.increment_E(s,a)
        for i in range(self.action_space):
            for j in range(self.state_space):
                self.Q[j][i] = self.Q[j][i] + self.ALPHA * delta * self.E[j][i]
                self.E[j][i] = self.GAMMA * self.LAMBDA * self.E[j][i]

    def clear_eligibilty_trace(self):
        self.E = [[0 for a in range(self.action_space)] for s in range(self.state_space)]