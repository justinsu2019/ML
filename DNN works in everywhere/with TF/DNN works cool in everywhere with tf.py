#Try to compare the difference bewteen every activation function
#when it comes to RNN, we will need to backward the inputs


import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import math
import random
import numpy as np
import excelExecuterFinalV3 as ee
import matplotlib.pyplot as plt
import easygui as eg
import pprint
from tensorflow.python import pywrap_tensorflow


### pre define start###
#put in our data -----------------------------------


### controller start ###
#123 stands for final, highest, lowest
target = 1 # select which sheetname is our targeted label, refer to ee; tips: 1 stand sheet 2
saveOrNot = 0
loadOrNot = 0
inputFile = 'EURD'
activation_function = "elu"


# Black magic, optimized 'if' function
label = ((target == 1 and ee.ee(inputFile,1).ds()) or (target == 2 and ee.ee(inputFile,2).ds()) or (target == 3 and ee.ee(inputFile,3).ds())) # if we need to change label, we change here

data = ee.ee(inputFile,0).ds()[:len(label)] # now, all our data is inputFile's sheet1 data.
test = np.array(ee.ee(inputFile,0).ds()[len(label)-1])[np.newaxis,:] # test data is the last line normally, of course we can change it if it's not.
saver_file = "Networksaver/"+inputFile+"weights.ckpt" # save file to this location


#when lr more than 0.05, it may goes to be bad
#best biases is 0.5 now, accuracy is 0.439767
data_size = len(ee.ee(inputFile,0).ds()[0])
hidden_size = 2*data_size
label_size = len(ee.ee(inputFile,1).ds()[0])
loop = 10000
lr = 0.01
Biases = 0.05

### controller end ###----------------------------------------------------------


#x_data is purely only for visilization good 
x_data = np.linspace(0.5,1.5,len(label))[:,np.newaxis]


#define activation function,
#till now, elu is the best in performance
string = str("tf.nn.")+str(activation_function) # make it fexiable to change the activation function
activation = exec(string) # 2 str won't be recognized by tf, so have to exec it


#define how the network looks like
### this is how we define a layer start ###
def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope("Layer"):
        with tf.name_scope("Weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope("Biases"):
            biases = tf.Variable(tf.zeros([1,out_size])+Biases)
        with tf.name_scope("Out"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            
        outputs = (activation_function is not None and (activation_function(Wx_plus_b)) or Wx_plus_b) # for tensow, use "is not None instead of is None"
        
        return outputs
### this is how we define a layer end ###------------------------------------------------------------------



#annouce the space for inputs and outputs
#name scope is for graph, for visilization
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None,data_size], name="Base_data")
    ys = tf.placeholder(tf.float32, [None,label_size], name="Label")


#build layers
l1 = add_layer(xs,data_size,hidden_size,activation)   # layer 1
prediction = add_layer(l1,hidden_size,label_size,None)  # output layer


#define loss
with tf.name_scope("Loss"):
    #get loss, but here we have % loss or true loss, the first line is true but we may need %
    print()
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices =[1]))
    #loss = tf.reduce_mean(tf.reduce_sum((ys-prediction)/ys))

#define train, we have many train models, which should be tested later
with tf.name_scope("Train"):
    #train_step = tf.train.GradientDescentOptimizer(0.09).minimize(loss)
    #the best Optimizer till now is Adam!
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

#define initial, we will need to run it soon before everyelse things carry out
init = tf.global_variables_initializer()
### pre define end###-------------------------------------------------------------------------------


'''
### make it visiable start ###
#set the window
fig = plt.figure()
#set how many windows and how many slices we need
ax = fig.add_subplot(1,1,1)
#setup coordinate system
ax.scatter(x_data, label)
#should be the same as plt.show(block=false), make it fresh each time.
plt.ion()
#show the whole thing, last step can be remove as long as we have plt.ion()
plt.show()
### make it visiable end ###
'''



#store the whole nn 1/3
saver = tf.train.Saver()


### start nn start ###

#with with sentence, we don't have to close session as it will be closed automatically
with tf.Session() as sess:

    #build graph start#
    #writer = tf.summary.FileWriter("Graph/",tf.get_default_graph())
    #writer.add_graph(sess.graph)
    #writer.close()
    #build graph end#

    #initial the whole nn
    sess.run(init)

    #store the whole nn 2/3
    (loadOrNot==1 and saver.restore(sess, saver_file) or 0)     # better coding for restoration


    #start training  --------------------------------------------------------------------------------------------------------------------------------------------------------------
    for i in range(loop):

        #input data into train function and let it work
        sess.run(train_step, feed_dict={xs: data, ys: label})

        #let the nn stop once the accuracy is more than 9.9999%
        if sess.run(loss, feed_dict={xs:data, ys: label}) < 0.0002:
            print("Aim is {}".format(sess.run(prediction, feed_dict={xs: test})[0][0]))
            break
        else:
            #print out every x times
            if i % 250 == 0:

                #need to remove the previous prediction line before draw new ones
                ###try:
                    ###ax.lines.remove(lines[0])   #because line has only 1 data, so it means clear the value in lines

                #if no lines had been drew before, as it's the 1st time, then pass
                ###except Exception:
                    ###pass

                #print out the accuracy

                prediction_value = sess.run(prediction, feed_dict={xs: data})

                #lines = ax.plot(data, prediction_value,'r-',lw=3)
                #lw means line width
                ###lines = ax.plot(x_data,prediction_value,'r-',lw=1)

                #print out, \n is for enter, and [0][0] is to select the very first value in whole matrix
                print("accuracy is {}".format(round(1-sess.run(loss, feed_dict={xs:data, ys: label}),6)))

                # because "xxx" is str, so if we want '+' instead of ',' (because use , there will be a enter
                # so we will need to str(the value)
                #below is for predict only 1 line:
                print(i," times, Aim is {}".format(str(sess.run(prediction, feed_dict={xs: test})[0][0]).strip()))

                #here is for batch test prediction
                
                #plt pause to make it more visiable
                #plt.pause(0.1)

                
    #store the whole nn 3/3
    (saveOrNot==1 and print("Save to path:", saver.save(sess,saver_file)) or 0)

### start nn end ###

eg.msgbox("Hey, Susu has the solution now!") # remind you once we have all work done!
