import numpy as np

class Filterer:
    def __init__(self,size):
        self.sample_vector_0 = np.zeros(size)
        self.sample_vector_1 = np.zeros(size)
        self.sample_vector_2 = np.zeros(size)
        self.state = 0
    
    def medianfilter(self, vec):
        if self.state==0:
            for x in range(0,len(self.sample_vector_0)):
                self.sample_vector_0[x] = vec[0]
            for x in range(0,len(self.sample_vector_1)):
                self.sample_vector_1[x] = vec[1]
            for x in range(0,len(self.sample_vector_2)):
                self.sample_vector_2[x] = vec[2]
            self.state = 1
            return vec

        for x in range(len(self.sample_vector_0)-1,0,-1):
            self.sample_vector_0[x] = self.sample_vector_0[x-1]
        for x in range(len(self.sample_vector_1)-1,0,-1):
            self.sample_vector_1[x] = self.sample_vector_1[x-1]
        for x in range(len(self.sample_vector_2)-1,0,-1):
            self.sample_vector_2[x] = self.sample_vector_2[x-1]

        self.sample_vector_0[0] = vec[0]
        self.sample_vector_1[0] = vec[1]
        self.sample_vector_2[0] = vec[2]
        vec[0] = np.median(self.sample_vector_0)
        vec[1] = np.median(self.sample_vector_1)
        vec[2] = np.median(self.sample_vector_2)

        return vec
    def meanfilter(self, vec):
        if self.state==0:
            for x in range(0,len(self.sample_vector_0)):
                self.sample_vector_0[x] = vec[0]
            for x in range(0,len(self.sample_vector_1)):
                self.sample_vector_1[x] = vec[1]
            for x in range(0,len(self.sample_vector_1)):
                self.sample_vector_2[x] = vec[2]
            self.state = 1
            return vec

        for x in range(len(self.sample_vector_0)-1,0,-1):
            self.sample_vector_0[x] = self.sample_vector_0[x-1]
        for x in range(len(self.sample_vector_1)-1,0,-1):
            self.sample_vector_1[x] = self.sample_vector_1[x-1]
        for x in range(len(self.sample_vector_2)-1,0,-1):
            self.sample_vector_2[x] = self.sample_vector_2[x-1]

        self.sample_vector_0[0] = vec[0]
        self.sample_vector_1[0] = vec[1]
        self.sample_vector_2[0] = vec[2]
        vec[0] = np.mean(self.sample_vector_0)
        vec[1] = np.mean(self.sample_vector_1)
        vec[2] = np.mean(self.sample_vector_2)

        return vec