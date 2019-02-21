from nWalk import nWalk
from Sarsa_Lambda import Sarsa_Lambda

def observation2state(observation):
    return observation.index(1)

size = 19
env = nWalk(size)
agent = Sarsa_Lambda(size,2,0.5,0.9)

for eps in range(100):
    observation = env.reset()
    total_reward = 0
    steps = 0
    done = 0
    s = observation2state(observation)
    a = agent.act(s)
    while(not done):
        observation, reward, done = env.step(a)
        #print(agent.Q)
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
    print()
    agent.clear_eligibilty_trace()

    if(eps % 10 == 0):
        agent.epsilion_step()

print("nWalk size 19 - Eligibilty Q-Table (State-Action Table)")
print("-"*60)
print(agent.Q)