import math
import random
import numpy as np
import excelExecuter as ee
from sklearn.model_selection import train_test_split #数据集分割


random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a

# create weights automatically
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


# define activity function start
def PReLU(x):
    return max(x,0.15 * x)

def de_PReLU(x):
    y = 0
    if x >= 0:
        y = 1
    else:
        y = 0.15
    return y


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)
# define activity function end



# create nn start 
class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):  #ni = n input, nh = n hidden, no = n output
        self.input_n = ni + 1    # add 1 basis in 
        self.hidden_n = nh
        self.output_n = no
        
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n

        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        
        # random activate 
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)

        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):  # also known as forward_propagate()
        
        # activate input layer
        # print(inputs)
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
            
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
            
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        
        # feed forward
        self.predict(case)
        output_deltas = [0.0] * self.output_n

        # get output layer error
        if isinstance(label,np.int64):
            #print('label is a number')
            for o in range(self.output_n):
                error = label - self.output_cells[o]
                output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        else:
            for o in range(self.output_n):
                error = label[o] - self.output_cells[o]
                output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error



        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error

        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change

        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change

        # get global error start
        error = 0.0
        if isinstance(label,np.int64):
            #print('label is a number')
            error += 0.5 * (label - self.output_cells) ** 2
        else:
            for o in range(len(label)):
                error += 0.5 * (label[o] - self.output_cells[o]) ** 2

        


        # check out correct start
        #if error < 0.0001:
            #print("Under",label,"there is ", error, "error")
        # check out correct end
        
        return error
        # get global error end


    def train(self, cases, labels, limit=1, learn=0.02, correct=0.1):
        for j in range(limit):
            error = 0.0
            for i in range(cases.shape[0]):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
            print('Time record ',j,', our error is',error)
        print(error)
        return error


    # main function start
    def test(self):

        case,label = ee.data_process()

        #print(case.shape[1], np.unique(label),label.shape)
        
        self.setup(case.shape[1], 2*case.shape[1], 1)
        self.train(case, label, 3, 0.02, 0.4)

        print(self.predict([0,0,1,0,1,0,0,0,1,0]))
        print(self.predict([0,1,1,1,1,0,1,0,1,1]))
    # main function end

# create nn end


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
