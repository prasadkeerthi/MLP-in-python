import numpy as np
import math

class MLP:
    def __init__(self, layers):
        self.shape = layers
        
        self.z1 = []

        self.z1.append(np.ones(self.shape[0]+1))
        n = len(layers)
        for i in range(1,n):
            self.z1.append(np.ones(self.shape[i]))

        self.Wt = []
        for i in range(n-1):
            self.Wt.append(np.zeros((self.z1[i].size,self.z1[i+1].size)))

        self.dw = [0]*len(self.Wt)
        self.adjust_Wt()


    def forward_pass(self, input_data):

        self.z1[0][0:-1] = input_data
        for i in range(1,len(self.shape)):
            self.z1[i] = self.tanh_activation(np.dot(self.z1[i-1],self.Wt[i-1]))
        return self.z1[-1]


    def backward_pass(self, Y, alpha=0.1, cr=0.1):

        difference_error = Y - self.z1[-1]
        diff = difference_error*self.delta_tanh(self.z1[-1])
        changes_array = []
        changes_array.append(diff)


        for i in range(len(self.shape)-2,0,-1):
            diff = np.dot(changes_array[0],self.Wt[i].T)*self.delta_tanh(self.z1[i])
            changes_array.insert(0,diff)
            

        for i in range(len(self.Wt)):
            z1 = np.atleast_2d(self.z1[i])
            diff = np.atleast_2d(changes_array[i])
            dw = np.dot(z1.T,diff)
            self.Wt[i] += alpha*dw + cr*self.dw[i]
            self.dw[i] = dw
        return abs(difference_error)
        
    
    def tanh_activation(self,val):
        return np.tanh(val)

    def delta_tanh(self,val):
        return 1.0-val**2

    def adjust_Wt(self):
        for i in range(len(self.Wt)):
            Z = np.random.random((self.z1[i].size,self.z1[i+1].size))
            self.Wt[i]= Z


def train_XOR(model,input_dataset, epochs=2000, alpha=.2, lr=0.1):

    for i in range(epochs):
        n = np.random.randint(input_dataset.size)
        model.forward_pass( input_dataset['input'][n] )
        #print(input_dataset['input'][n].shape)
        model.backward_pass( input_dataset['output'][n], alpha, lr )

    for i in range(input_dataset.size):
        o = model.forward_pass( input_dataset['input'][i] ) 
        print (i, input_dataset['input'][i], '%.2f' % o[0])
        print ('(expected %.2f)' % input_dataset['output'][i])

layers=[2,3,3,1]
xor_model = MLP(layers)
input_dataset = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])




xor_model.adjust_Wt()
input_dataset[0] = (0,0), 0
input_dataset[1] = (1,0), 1
input_dataset[2] = (0,1), 1
input_dataset[3] = (1,1), 0
train_XOR(xor_model, input_dataset)






print ("Learning the sin function")
layers=[1,10,10,10,10,10,1]
sin_model = MLP(layers)
input_dataset = np.zeros(200, dtype=[('x1',  float, 1),('x2',  float, 1),('x3',  float, 1),('x4',  float, 1), ('y', float, 1)])
input_dataset['x1'] = np.random.uniform(-1,1,200) 
input_dataset['x2'] = np.random.uniform(-1,1,200) 
input_dataset['x3'] = np.random.uniform(-1,1,200) 
input_dataset['x4'] = np.random.uniform(-1,1,200) 
input_dataset['y'] = np.sin(input_dataset['x1']-input_dataset['x2']+input_dataset['x3']-input_dataset['x4'])

print("Training on 150 rows\n")
for k in range(1000):
    for i in range(150):
        
        inputarray=np.asarray([input_dataset['x1'][i]-input_dataset['x2'][i]+input_dataset['x3'][i]-input_dataset['x4'][i]]) 
        
        av=sin_model.forward_pass(inputarray)
        sin_model.backward_pass(input_dataset['y'][i],0.2)
        if k%100==0 and i==0:
            print("mae is ",np.absolute(np.subtract(input_dataset['y'][i], av)).mean())
        
        
total_difference_error=0         
for i in range(149,200):
    
    av=sin_model.forward_pass([input_dataset['x1'][i]-input_dataset['x2'][i]+input_dataset['x3'][i]-input_dataset['x4'][i]]) 
    total_difference_error+=np.absolute(np.subtract(input_dataset['y'][i], av).mean())

print("mean absolute difference_error after testing for 50 rows  is",total_difference_error/50)
    


