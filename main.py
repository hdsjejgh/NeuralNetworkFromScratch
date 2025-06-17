import csv
import random
import math


Xs = []
Ys = []
EPOCHS = 15
NUM_FEATURES = 2
LEARNING_RATE = 0.1

with open("synthetic_data.csv","r") as f:
    file = csv.reader(f)
    next(file)
    for line in file:
        line = list(map(float,line))
        Xs.append([[x] for x in line[:-1]])
        Ys.append(line[-1])


NUM_SAMPLES = len(Xs)
BATCH_SIZE = 1

ALPHA = 0.1
def LReLu(v):
    return [i if i[0]>0 else [ALPHA*ii for ii in i] for i in v]

def LReLuP(v): #Derivative of relu function
    return [[1] if i[0]>0 else [ALPHA] for i in v]


def sigmoid(n):
    if n[0][0]>500:
        return 0.99999
    if n[0][0]<-500:
        return 0.00001
    return 1/(1+math.exp(-n[0][0]))



#Network Shape is 2,4,1
layers = [2,4,1]
activations= [None, LReLu, sigmoid]

#Initialized weights in the range [-1/sqrt d, 1/sqrt d], d is number of inputs
weights = [
    [
        [
            random.uniform(-1/math.sqrt(layers[i]),1/math.sqrt(layers[i]))
            for iii in range(layers[i])
        ]  for ii in range(layers[i+1])
    ]    for i in range(len(layers)-1)
]

biases = [[[0] for ii in range(i)] for i in layers[1:]]


def T(m): #returns transpose of a matrix
    rows = len(m)
    cols = len(m[0])
    transpose = [[m[i][ii] for i in range(rows)] for ii in range(cols)]
    return transpose

def empty(h,w): #returns empty matrix of given width and height
    return [[0 for i in range(w)] for ii in range(h)]


def inner_prod(m1,m2): #inner product between matrices
    assert len(m1[0])==len(m2), f"Invalid matrix dimensions {len(m1)}x{len(m1[0])} and {len(m2)}x{len(m2[0])}"
    m2 = T(m2)
    h,w = len(m1), len(m2)
    product = empty(h,w)
    for i in range(w):
        for ii in range(h):
            v1 = m1[ii]
            v2 = m2[i]
            p = sum(v1[i]*v2[i] for i in range(len(v1)))
            product[ii][i]=p
    return product

def mat_op(m1,m2,operator): #Elementwise operations between 2 matrices
    assert len(m1)==len(m2) and len(m1[0])==len(m2[0]), f"Invalid matrix dimensions {len(m1)}x{len(m1[0])} and {len(m2)}x{len(m2[0])}"
    h,w = len(m1),len(m1[0])
    for i in range(h):
        for ii in range(w):
            m1[i][ii]=eval(f"m1[i][ii]{operator}m2[i][ii]")
    return m1

def mat_scale(m,scalar):
    for i in range(len(m)):
        for ii in range(len(m[0])):
            m[i][ii]*=scalar
    return m

def forward(inp): #Forward propagation, returns prediction, and activated parts
    As = [inp] #Activated parts
    for i in range(1,len(layers)):

        inp = inner_prod(weights[i-1],inp)
        inp = mat_op(inp,biases[i-1],"+")
        inp = activations[i](inp)
        As.append(inp)

    return (inp,As)


for i in range(EPOCHS):

    indices = list(range(NUM_SAMPLES))
    random.shuffle(indices)
    for ii in indices:
        x, y = Xs[ii], Ys[ii]
        pred,As = forward(x)
        dL_dz3 = pred-y

        #Layer 2
        dL_dw2 = mat_scale(T(As[1]),dL_dz3)
        dL_db2 = [[dL_dz3]]


        #Layer 1
        dL_da2 = inner_prod(T(weights[1]), [[dL_dz3]])
        activation_derivs = LReLuP(As[1])
        dL_dz2 = mat_op(dL_da2, activation_derivs, "*")

        dL_db1 = dL_dz2
        dL_dw1 = inner_prod(dL_dz2, T(As[0]))

        weights[1] = mat_op(weights[1], mat_scale(dL_dw2, LEARNING_RATE/BATCH_SIZE), "-")
        biases[1] = mat_op(biases[1], mat_scale(dL_db2, LEARNING_RATE/BATCH_SIZE), "-")
        weights[0] = mat_op(weights[0], mat_scale(dL_dw1, LEARNING_RATE/BATCH_SIZE), "-")
        biases[0] = mat_op(biases[0], mat_scale(dL_db1, LEARNING_RATE/BATCH_SIZE), "-")


correct = 0
for i in range(NUM_SAMPLES):
    correct+=1 if round(forward(Xs[i])[0])==Ys[i] else 0
print(f"Accuracy: {100*correct/NUM_SAMPLES:.2f}%")

