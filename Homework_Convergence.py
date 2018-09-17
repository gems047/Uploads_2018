import numpy as np

X = np.array([
    [1,1],
    [-1,-1],
    [2,2],
    [-2,-2],
    [-1, 1],
    [1,-1]

])

w = [1,0.0]
wnew=[0,0]
eta = 0.001
epochs = 300

T = np.array([-1,-1,1,-1,1,1])
O = np.array([0.0,0.0, 0.0,0.0,0.0,0.0])    

y = np.array([1,-1,1,-1,1,1])
#Error =np.array([0,0]) 
# A = np.array([
#              [0,0],
#              [0,0],
#              [0,0],
#              [0,0],
#              [0,0],
#              [0,0]
#              ])

#A1 = np.array([0,0,0,0,0,0])
#A2 = np.array([0,0,0,0,0,0])
X1 = [x[0] for x in X]
X2 = [x[1] for x in X]
#print(X1)
#print(X2)

def batch_perceptron(X,Y):
    global w
    for t in range(epochs):
        for i,value in enumerate(X):
            O[i] = np.dot(w,X[i])
            #print(O[i])
            if(O[i] >= 0):
                O[i]=1
            if(O[i] < 0):
                O[i]=-1
            print(O[i])   

    #A1 = (T-O)*X1
    #A2 = (T-O)*X2
    #print(A1)
    #print(A2)   
    
    
    
        t1 = np.dot((T-O),X1)
        e1= eta*t1
        #print(w[0],e1)
        w[0] = w[0]+e1
        t2 = np.dot((T-O),X2)
        e2= eta*t2

        
        w[1] = w[1]+e2
        print(w)
    return w
    
w = batch_perceptron(X,y)
#print(w)



