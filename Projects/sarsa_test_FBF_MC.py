import gym
import numpy as np
from Sarsa_Lambda_FBF import Sarsa_Lambda

data = []

def observation2state(observation,low,high):
    return (observation-low)/(high-low)

env = gym.make('MountainCar-v0')
env._max_episode_steps = 10000
low, high = env.observation_space.low, env.observation_space.high
agent = Sarsa_Lambda(env.observation_space.shape[0],env.action_space.n,0.9,1,EPSILON_START=1)

for eps in range(20):
    observation = env.reset()
    total_reward = 0
    steps = 0
    done = 0
    s = observation2state(observation,low,high)
    a = agent.act(s)
    while(not done):
        env.render()
        observation, reward, done, _ = env.step(a)
        #print(observation)
        #print(agent.Q.w)
        #print(agent.E)
        #input("Next?")
        _s = observation2state(observation,low,high)
        _a = agent.act(_s)
        
        agent.update(s,a,reward,_s,_a)
        
        s = _s
        a = _a

        steps += 1
        total_reward += reward
    print("Episode",eps+1,"finished after",steps,"steps")
    print("Total reward was",total_reward)
    print("EPSILON:",agent.EPSILON)
    print()
    data.append(steps)
    agent.clear_eligibilty_trace()

    if(eps % 1 == 0):
        agent.epsilion_step()

#print(agent.Q.w)
#print(agent.EPSILON)

import matplotlib.pyplot as plt


x = data.copy()
y = [i+1 for i in range(len(x))]

plt.plot(y, x)
plt.show()