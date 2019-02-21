import gym
import numpy as np
from Sarsa_Lambda_FBF import Sarsa_Lambda

data = []

def observation2state(observation,low,high):
    return (observation-low)/(high-low)

env = gym.make('CartPole-v1')
env._max_episode_steps = 10000
low, high = env.observation_space.low, env.observation_space.high
agent = Sarsa_Lambda(env.observation_space.shape[0],env.action_space.n,0.9,0.9,EPSILON_START=0.7,EPSILON_STEP=0.005,order=5)

total_reward = 0
steps = 0
n = 10
for eps in range(1000):
    observation = env.reset()
    done = 0
    s = observation2state(observation,low,high)
    a = agent.act(s)
    while(not done):
        env.render()
        observation, reward, done, _ = env.step(a)
        #print(observation)
        #print(agent.Q.compute(s))
        #print(agent.E)
        #input("Next?")
        _s = observation2state(observation,low,high)
        _a = agent.act(_s)
        
        agent.update(s,a,reward,_s,_a)
        
        s = _s
        a = _a

        steps += 1
        total_reward += reward
    agent.clear_eligibilty_trace()

    if((eps + 1) % n == 0):
        data.append(steps/n)
        print("Episode finished after",steps/n,"steps")
        print("Total reward was",total_reward/n)
        print("Epsilon:",agent.EPSILON)
        print()
        total_reward = 0
        steps = 0
        agent.epsilion_step()

print("Epsilon:")
print(agent.EPSILON)

import matplotlib.pyplot as plt


x = data.copy()
y = [i+1 for i in range(len(x))]

plt.plot(y, x)
plt.show()