from nWalk import nWalk
from Sarsa_Lambda_FBF import Sarsa_Lambda

def observation2state(observation):
    return observation.index(1)/(len(observation) + 1)

size = 19
env = nWalk(size)
agent = Sarsa_Lambda(1,2,0.5,0.9,1)

for eps in range(500):
    observation = env.reset()
    total_reward = 0
    steps = 0
    done = 0
    s = observation2state(observation)
    a = agent.act(s)
    while(not done):
        observation, reward, done = env.step(a)
        #print(agent.Q.w)
        #print(agent.E)
        #input("Next?")
        _s = observation2state(observation)
        _a = agent.act(_s)
        
        agent.update(s,a,reward,_s,_a)
        
        s = _s
        a = _a

        steps += 1
        total_reward += reward
    print("Episode",eps+1,"finished after",steps,"steps")
    print("Total reward was",total_reward)
    agent.clear_eligibilty_trace()

    if(eps % 10 == 0):
        agent.epsilion_step()

print()
print("nWalk size 19 - Fourier Basis Weights => order=5 state_space_size=1 action_space_size=2")
print("-"*60)
print(agent.Q.w)