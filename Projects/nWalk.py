class nWalk():
    def __init__(self,n,max_steps=50,start=None):
        self.n = n
        self.max_steps = max_steps
        self.start = start

        self.reset()

    def show(self):
        print(" ".join(self.line))

    def step(self,a):
        done = 0
        reward = 0
        observation = []

        if(a == 0):
            s = self.line[self.agent_pos - 1]
            self.line[self.agent_pos - 1] = "X"
            self.line[self.agent_pos] = s
            self.agent_pos = self.agent_pos - 1
        elif(a == 1):
            s = self.line[self.agent_pos + 1]
            self.line[self.agent_pos + 1] = "X"
            self.line[self.agent_pos] = s
            self.agent_pos = self.agent_pos + 1
        else:
            raise Exception('Invalid action ... action expected to be 0 or 1')


        for element in self.line:
            if(element == "-"):
                observation.append(3)
            elif(element == "+"):
                observation.append(2)
            elif(element == "X"):
                observation.append(1)
            elif(element == "_"):
                observation.append(0)

        if(self.agent_pos == 0):
            reward = -1
            done = 1
        elif(self.agent_pos == (self.n - 1)):
            reward = 1
            done = 1
        elif(self.steps >= (self.max_steps - 1)):
            done = 1

        self.steps = self.steps + 1
        return observation, reward, done

    def reset(self):
        observation = []
        self.steps = 0

        self.line = ["_" for i in range(self.n)]
        self.line[0] = "-"
        self.line[-1] = "+"
        
        if(self.start == None):
            self.agent_pos = int(self.n/2)
            self.line[self.agent_pos] = "X"
        elif(self.start >= self.n - 1 or self.start <= 0):
            raise Exception('Start position has to be less than n - 1 and greater than 0')
        else:
            self.agent_pos = int(self.start)
            self.line[self.agent_pos] = "X"

        for element in self.line:
            if(element == "-"):
                observation.append(3)
            elif(element == "+"):
                observation.append(2)
            elif(element == "X"):
                observation.append(1)
            elif(element == "_"):
                observation.append(0)

        return observation