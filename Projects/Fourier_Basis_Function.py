import numpy as np 
import math   

class Fourier_Basis_Function():
    def __init__(self,input_size,output_size,order,ALPHA):
        self.order = order
        self.input_size = input_size
        self.c = np.array(self.enumerate_c(input_size,order))
        self.w = np.array([[0.5 for i in range((order + 1)**input_size)] for a in range(output_size)])
        self.size = (order + 1)**input_size
        self.ALPHA = ALPHA


        
        self.scaling = np.linalg.norm(self.c, axis = 1)
        self.scaling[self.scaling==0.] = 1.
        self.scaling = 1./self.scaling

    def enumerate_c(self,input_size,order):
        a = [None for i in range((order + 1)**input_size)]
        curr = [0 for i in range(input_size)]
        a[0] = curr.copy()
        for i in range(1,(order + 1)**input_size):
            incremented = False
            for index in range(input_size):
                if(curr[index] <= order and incremented == False):
                    curr[index] = curr[index] + 1
                    incremented = True
                if(curr[index] > order and index + 1 != input_size):
                    curr[index] = 0
                    curr[index + 1] = curr[index + 1] + 1
                else:
                    break
            a[i] = curr.copy()
        return a

    def compute(self,x):
        return np.dot(self.w,np.cos(np.multiply(math.pi,np.dot(self.c, np.array(x)))))

    def gradient_step(self,x,target):
        delta = self.compute(x)
        diff = self.ALPHA * delta * -1 * (np.subtract(np.array(target),np.array(delta)))
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - diff[i]
        print(0.5 * np.subtract(np.array(target),np.array(delta))**2)
