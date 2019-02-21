import random
import numpy as np
from Fourier_Basis_Function import Fourier_Basis_Function

class Sarsa_Lambda():
    def __init__(self,state_space,action_space,LAMBDA,GAMMA,EPSILON_START,EPSILON_MAX=0.99,EPSILON_STEP=0.01,order=5,method="Fourier"):
        self.state_space = state_space
        self.action_space = action_space
        self.ALPHA = 0.001 #0.001
        self.LAMBDA = LAMBDA
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON_START
        self.EPSILON_MAX = EPSILON_MAX
        self.EPSILON_STEP = EPSILON_STEP
        self.method = method

        # Action-Value Function
        self.Q = Fourier_Basis_Function(state_space,action_space,order,0.1)

        # Eligability Trace
        self.E = np.zeros_like(self.Q.w)

    def act(self,state):
        action = None
        rand = random.randint(1,101)
        if(rand < (self.EPSILON * 100)):
            action = np.argmax(self.Q.compute(state))
        else:
            action = random.randint(0,self.action_space-1)
        return action

    def epsilion_step(self):
        if(self.EPSILON != self.EPSILON_MAX):
            self.EPSILON = self.EPSILON + self.EPSILON_STEP
            if(self.EPSILON > self.EPSILON_MAX):
                self.EPSILON = self.EPSILON_MAX 

    def calc_delta(self,s,a,r,_s,_a):
        return r + self.GAMMA * self.Q.compute(_s)[_a] - self.Q.compute(s)[a]

    def increment_E(self,a):
        self.E[a] = [e + 1 for e in self.E[a]]

    def update(self,s,a,r,_s,_a):
        delta = self.calc_delta(s,a,r,_s,_a)
        self.increment_E(a)
        self.Q.w = np.add(self.Q.w,self.ALPHA * delta * np.tile(self.Q.scaling, (self.action_space,1)) * np.array(self.E))
        self.E = self.GAMMA * self.LAMBDA * np.array(self.E)

    def clear_eligibilty_trace(self):
        self.E = np.zeros_like(self.Q.w)