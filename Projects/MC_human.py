import gym
import numpy as np
from Sarsa_Lambda_FBF import Sarsa_Lambda
import keyboard

data = []

env = gym.make('MountainCar-v0')
env._max_episode_steps = 10000
low, high = env.observation_space.low, env.observation_space.high

print("Use A and D keys to move cart!!!")
for eps in range(1000):
    observation = env.reset()
    total_reward = 0
    steps = 0
    done = 0
    a = 1
    while(not done):
        env.render()
        try: #used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('a'):#if key 'q' is pressed 
                print('You Pressed a Key!')
                a = 0
            if keyboard.is_pressed('s'):#if key 'q' is pressed 
                print('You Pressed s Key!')
                a = 1
            if keyboard.is_pressed('d'):#if key 'q' is pressed 
                print('You Pressed d Key!')
                a = 2
            else:
                pass
        except:
            pass
        observation, reward, done, _ = env.step(a)

        steps += 1
        total_reward += reward
    print("Episode finished after",steps,"steps")
    print("Total reward was",total_reward)
    data.append(steps)

import matplotlib.pyplot as plt


x = data.copy()
y = [i+1 for i in range(len(x))]

plt.plot(y, x)
plt.show()