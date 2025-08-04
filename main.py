import csv
import random
import math
import matplotlib.pyplot as plt



Xs = []
Ys = []
EPOCHS = 12
NUM_FEATURES = 20
NUM_CLASSES = 4
LEARNING_RATE = 0.001
BETA_RMS = 0.999 #Beta parameter for RMS prop part of ADAM
BETA_M = 0.9 #Beta parameter for momentum part of ADAM
EPSILON = 1e-8 #Adding in denominator of update to stop division by 0



with open("synthetic_multiclass_data.csv","r") as f:
    file = csv.reader(f)
    next(file)
    for line in file:
        line = list(map(float,line))
        Xs.append([[x] for x in line[:-1]])
        Ys.append(line[-1])

# print(set(Ys))

NUM_SAMPLES = len(Xs)
BATCH_SIZE = 100
TEST = .2 #Proportion of examples to be in test set
RECORD = True #Whether or not to record accuracy throughout epochs

test_ids = set(random.sample(list(range(NUM_SAMPLES)),int(NUM_SAMPLES*TEST)))
Xtest,Ytest = [],[]
Xtrain,Ytrain = [],[]
for i in range(NUM_SAMPLES):
    if i in test_ids:
        Xtest.append(Xs[i])
        Ytest.append(Ys[i])
    else:
        Xtrain.append(Xs[i])
        Ytrain.append(Ys[i])

ALPHA = 0.1 #Alpha for leaky ReLU
def LReLu(v):
    return [i if i[0]>0 else [ALPHA*ii for ii in i] for i in v]

def LReLuP(v): #"Leaky Relu Prime" Derivative of relu function
    return [[1] if i[0]>0 else [ALPHA] for i in v]

def sigmoid(n): #Sigmoid activation function (not used)
    #To stop overflow from really big or small numbers
    if n[0][0]>500:
        return 0.99999
    if n[0][0]<-500:
        return 0.00001
    return 1/(1+math.exp(-n[0][0]))

def Softmax(x): #softmax activation
    # print(x)

    #to stop overflow from big values
    max_val = max(i[0] for i in x)

    #sum for denominator
    s = sum(math.exp(i[0] - max_val) for i in x)

    for i in range(len(x)):
        num = x[i][0]
        num = math.exp(num-max_val)/s
        x[i]=[num]

    return x


layers = [NUM_FEATURES,10,5,NUM_CLASSES]
activations= [None, LReLu, LReLu, Softmax]

#Initialized weights in the range [-1/sqrt d, 1/sqrt d], d is number of inputs (xavier initialization)
weights = [
    [
        [
            random.uniform(-1/math.sqrt(layers[i]),1/math.sqrt(layers[i]))
            for iii in range(layers[i])
        ]  for ii in range(layers[i+1])
    ]     for i in range(len(layers)-1)
]

biases = [[[0] for ii in range(i)] for i in layers[1:]]




def T(m): #returns transpose of a matrix
    rows = len(m)
    cols = len(m[0])
    transpose = [[m[i][ii] for i in range(rows)] for ii in range(cols)]
    return transpose

def empty(h,w): #returns empty matrix of given width and height
    return [[0 for i in range(w)] for ii in range(h)]

def copy_matrix(m): #makes hard copy of matrix because python passes by reference for some godawful reason
    return [[m[i][j] for j in range(len(m[i]))] for i in range(len(m))]

def inner_prod(m1,m2): #inner product between matrices
    assert len(m1[0])==len(m2), f"Invalid matrix dimensions {len(m1)}x{len(m1[0])} and {len(m2)}x{len(m2[0])}" #Error if invalid matrix dimensions
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

def mat_op(m1,m2,operator): #performs elementwise operation between two matrices
    m1 = copy_matrix(m1) #copy as to not alter the original matrix
    assert len(m1)==len(m2) and len(m1[0])==len(m2[0]), f"Invalid matrix dimensions {len(m1)}x{len(m1[0])} and {len(m2)}x{len(m2[0])}"
    h,w = len(m1),len(m1[0])
    for i in range(h):
        for ii in range(w):
            m1[i][ii]=eval(f"m1[i][ii]{operator}m2[i][ii]")
    return m1

def mat_scale(m,scalar): #returns matrix scaled by a factor
    result = copy_matrix(m)
    for i in range(len(result)):
        for ii in range(len(result[0])):
            result[i][ii]*=scalar
    return result

def forward(inp): #Forward propagation, returns prediction, activated parts, and linear parts
    inp = copy_matrix(inp)
    #both activated and linear parts start with the input just for convenience
    As = [inp] #Activated parts
    Zs = [inp] #linear parts
    for i in range(1,len(layers)):
        #goes through each layer and applies activation
        inp = inner_prod(weights[i-1],inp)
        inp = mat_op(inp,biases[i-1],"+")
        Zs.append(inp)
        inp = activations[i](inp)
        As.append(inp)

    return (inp,As,Zs)

def adam_update(v,s): #returns the ADAM based update using the momentum (v) and rms (s) components
    #v/(sqrt(s)+epsilon)
    denom = [[math.sqrt(s[i][ii]) + EPSILON for ii in range(len(s[0]))] for i in range(len(s))]
    return mat_op(v, denom, '/')


def norm(m): #l2 norm of matrix
    return sum(sum(j**2 for j in i) for i in m)**.5




#records training and testing set accuracies if RECORD is true
train_acc = []
test_acc = []

#The momentum and rmsprop weights initialized to 0 for ADAM
vweights = [empty(layers[i+1],layers[i]) for i in range(len(layers)-1)]
sweights = [empty(layers[i+1],layers[i]) for i in range(len(layers)-1)]

#The momentum and rmsprop biases initialized to 0 for ADAM
vbiases = [[[0] for ii in range(i)] for i in layers[1:]]
sbiases = [[[0] for ii in range(i)] for i in layers[1:]]

#tests and records test set and train set accuracies to the train_acc and test_acc lists
def rec():
    correct = 0
    for i in range(len(Xtrain)):
        pred, _, _ = forward(Xtrain[i])
        pred = [i[0] for i in pred]
        correct += 1 if pred.index(max(pred)) == Ytrain[i] else 0
    train_acc.append(correct / len(Xtrain))

    correct = 0
    for i in range(len(Xtest)):
        pred, _, _ = forward(Xtest[i])
        pred = [i[0] for i in pred]
        correct += 1 if pred.index(max(pred)) == Ytest[i] else 0
    test_acc.append(correct / len(Xtest))

if RECORD: #initial recording of accuracies before first iteration
    rec()

for i in range(1,EPOCHS+1):
    print(f"Epoch: {i}")

    #not an efficient way of getting random indices but its good enough
    indices = list(range(len(Xtrain)))
    random.shuffle(indices)
    for iii,ii in enumerate(indices[:BATCH_SIZE]):
        x, y = Xtrain[ii], Ytrain[ii]

        #converts y into a vertical one hot vector
        ytemp = [[0]] * NUM_CLASSES
        ytemp[int(y)] = [1]
        y = ytemp

        pred, As, Zs = forward(x)


        # All vectors are reall dLoss/d-, but with dLoss/ omitted


        # ----Layer 3----

        dz3 = mat_op(pred, y, '-')

        dw3 = inner_prod(dz3, T(As[2]))
        db3 = dz3

        #calculates new momentum and rms weights
        vweights[2] = mat_op(mat_scale(vweights[2], BETA_M), mat_scale(dw3, 1 - BETA_M), '+')
        sweights[2] = mat_op(mat_scale(sweights[2], BETA_RMS), mat_scale(mat_op(dw3, dw3, '*'), 1 - BETA_RMS), '+')
        #bias correction
        vw3c = mat_scale(vweights[2], 1 / (1 - BETA_M ** i))
        sw3c = mat_scale(sweights[2], 1 / (1 - BETA_RMS ** i))

        #calculates new momentum and rms biases
        vbiases[2] = mat_op(mat_scale(vbiases[2], BETA_M), mat_scale(db3, 1 - BETA_M), '+')
        sbiases[2] = mat_op(mat_scale(sbiases[2], BETA_RMS), mat_scale(mat_op(db3, db3, '*'), 1 - BETA_RMS), '+')
        # bias correction
        vb3c = mat_scale(vbiases[2], 1 / (1 - BETA_M ** i))
        sb3c = mat_scale(sbiases[2], 1 / (1 - BETA_RMS ** i))

        #updates bias vector
        biases[2] = mat_op(biases[2], mat_scale(adam_update(vb3c, sb3c), LEARNING_RATE / 1), '-')

        # ----Layer 2----

        da2 = inner_prod(T(weights[2]), dz3)
        dz2 = mat_op(da2, LReLuP(Zs[2]), '*')

        dw2 = inner_prod(dz2, T(As[1]))
        db2 = dz2

        # calculates new momentum and rms weights
        vweights[1] = mat_op(mat_scale(vweights[1], BETA_M), mat_scale(dw2, 1 - BETA_M), '+')
        sweights[1] = mat_op(mat_scale(sweights[1], BETA_RMS), mat_scale(mat_op(dw2, dw2, '*'), 1 - BETA_RMS), '+')
        # bias correction
        vw2c = mat_scale(vweights[1], 1 / (1 - BETA_M ** i))
        sw2c = mat_scale(sweights[1], 1 / (1 - BETA_RMS ** i))

        # calculates new momentum and rms biases
        vbiases[1] = mat_op(mat_scale(vbiases[1], BETA_M), mat_scale(db2, 1 - BETA_M), '+')
        sbiases[1] = mat_op(mat_scale(sbiases[1], BETA_RMS), mat_scale(mat_op(db2, db2, '*'), 1 - BETA_RMS), '+')
        # bias correction
        vb2c = mat_scale(vbiases[1], 1 / (1 - BETA_M ** i))
        sb2c = mat_scale(sbiases[1], 1 / (1 - BETA_RMS ** i))

        #updates bias vector
        biases[1] = mat_op(biases[1], mat_scale(adam_update(vb2c, sb2c), LEARNING_RATE / 1), '-')

        # ----Layer 1----

        da1 = inner_prod(T(weights[1]), dz2)
        dz1 = mat_op(da1, LReLuP(Zs[1]), '*')

        dw1 = inner_prod(dz1, T(As[0]))
        db1 = dz1

        # calculates new momentum and rms weights
        vweights[0] = mat_op(mat_scale(vweights[0], BETA_M), mat_scale(dw1, 1 - BETA_M), '+')
        sweights[0] = mat_op(mat_scale(sweights[0], BETA_RMS), mat_scale(mat_op(dw1, dw1, '*'), 1 - BETA_RMS), '+')
        # bias correction
        vw1c = mat_scale(vweights[0], 1 / (1 - BETA_M ** i))
        sw1c = mat_scale(sweights[0], 1 / (1 - BETA_RMS ** i))

        # calculates new momentum and rms biases
        vbiases[0] = mat_op(mat_scale(vbiases[0], BETA_M), mat_scale(db1, 1 - BETA_M), '+')
        sbiases[0] = mat_op(mat_scale(sbiases[0], BETA_RMS), mat_scale(mat_op(db1, db1, '*'), 1 - BETA_RMS), '+')
        # bias correction
        vb1c = mat_scale(vbiases[0], 1 / (1 - BETA_M ** i))
        sb1c = mat_scale(sbiases[0], 1 / (1 - BETA_RMS ** i))

        #updates bias vector
        biases[0] = mat_op(biases[0], mat_scale(adam_update(vb1c, sb1c), LEARNING_RATE / 1), '-')

        if iii == 10: #Displays norm of dws for debugging purposes once every epoch
            print(f"Weights 3 Gradient: {adam_update(vw3c, sw3c)}")
            print(f"Weights 3 Gradient Norm: {norm(adam_update(vw3c, sw3c))}")
            print(f"Weights 2 Gradient: {adam_update(vw2c, sw2c)}")
            print(f"Weights 2 Gradient Norm: {norm(adam_update(vw2c, sw2c))}")
            print(f"Weights 1 Gradient: {adam_update(vw1c, sw1c)}")
            print(f"Weights 1 Gradient Norm: {norm(adam_update(vw1c, sw1c))}")

        #weights updated at the end so the updated ones don't mess with the backprop of other components
        weights[2] = mat_op(weights[2], mat_scale(adam_update(vw3c, sw3c), LEARNING_RATE / 1), '-')
        weights[1] = mat_op(weights[1], mat_scale(adam_update(vw2c, sw2c), LEARNING_RATE / 1), '-')
        weights[0] = mat_op(weights[0], mat_scale(adam_update(vw1c, sw1c), LEARNING_RATE / 1), '-')

    if RECORD: #record accuracies if enabled
        rec()

if RECORD: #Plots accuracy over time if enabled
    plt.plot([i for i in range(0,EPOCHS+1)],train_acc,color='red',label='Training Set Accuracy')
    plt.plot([i for i in range(0,EPOCHS+1)],test_acc,color='blue',label='Test Set Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Training Set and Testing Set Accuracy")
    plt.legend()

    plt.show()

#displays final accuracy
if RECORD: #if already recorded, just uses that one
    print(f"Test Accuracy: {100 * test_acc[-1]:.2f}%")
else:
    correct = 0
    for i in range(len(Xtest)):
        pred,_,_ = forward(Xtest[i])
        pred = [i[0] for i in pred]
        correct+=1 if pred.index(max(pred))==Ytest[i] else 0
    print(f"Test Accuracy: {100*correct/len(Xtest):.2f}%")


