#Try to compare the difference bewteen every activation function
#when it comes to RNN, we will need to backward the inputs

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import math
import random
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import excelExecuter as ee
import matplotlib.pyplot as plt
import easygui as eg
import pprint
import pandas as pd



### pre define start###
#put in our data -----------------------------------


### controller start ###
#123 stands for final, highest, lowest
target = 1 # select which sheetname is our targeted label, refer to ee; tips: 1 stand sheet 2
saveOrNot = 1
loadOrNot = 1
inputFile = 'Data_reading'
activation_function = "elu"



# Black magic, optimized 'if' function
label = ((target == 1 and ee.ee(inputFile,1).ds()) or (target == 2 and ee.ee(inputFile,2).ds()) or (target == 3 and ee.ee(inputFile,3).ds())) # if we need to change label, we change here
inputs_name = pd.read_excel(inputFile+".xlsx" ).columns
data = ee.ee(inputFile,0).ds()[:len(label)] # now, all our data is inputFile's sheet1 data.
test = np.array(ee.ee(inputFile,0).ds()[len(label)-1])[np.newaxis,:] # test data is the last line normally, of course we can change it if it's not.
saver_file = "Networksaver/"+inputFile+"weights.ckpt" # save file to this location
#print("data and label are {} \n and {} \n".format(data, label))


#when lr more than 0.05, it may goes to be bad
#best biases is 0.5 now, accuracy is 0.439767
data_size = len(ee.ee(inputFile,0).ds()[0])
hidden_size = 2*data_size
label_size = len(ee.ee(inputFile,1).ds()[0])
loop = 1000
lr = 0.005
Biases = 0.02

### controller end ###----------------------------------------------------------



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
        
        return outputs, Weights
### this is how we define a layer end ###------------------------------------------------------------------



#annouce the space for inputs and outputs
#name scope is for graph, for visilization
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None,data_size], name="Base_data")
    ys = tf.placeholder(tf.float32, [None,label_size], name="Label")


#build layers
l1,weight1 = add_layer(xs,data_size,hidden_size,activation)   # layer 1
prediction,weight2 = add_layer(l1,hidden_size,label_size,None)  # output layer


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


#store the whole nn 1/3
saver = tf.train.Saver()

### start nn start ###

#with with sentence, we don't have to close session as it will be closed automatically
with tf.Session() as sess:

    #initial the whole nn
    sess.run(init)

    #store the whole nn 2/3
    (loadOrNot==1 and saver.restore(sess, saver_file) or "")     # better coding for restoration


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

                prediction_value = sess.run(prediction, feed_dict={xs: data})

                #print out, \n is for enter, and [0][0] is to select the very first value in whole matrix
                print("accuracy is {}".format(round(sess.run(loss, feed_dict={xs:data, ys: label}),6)))

                # because "xxx" is str, so if we want '+' instead of ',' (because use , there will be a enter
                # so we will need to str(the value)
                #below is for predict only 1 line:
                print(i," times, Aim is {}".format(str(sess.run(prediction, feed_dict={xs: test})[0][0]).strip()))

    #store the whole nn 3/3

    (saveOrNot==1 and print("Save to path:", saver.save(sess,saver_file)) or 0)

    model_save_path = 'r"Networksaver/'+str(inputFile)+'weights.ckpt"'
    #model_reader = pywrap_tensorflow.NewCheckpointReader(model_save_path)
    model_reader = pywrap_tensorflow.NewCheckpointReader("Networksaver\Data_readingweights.ckpt")

    b = model_reader.get_tensor("Layer/Weights/Variable")
    c = model_reader.get_tensor("Layer_1/Weights/Variable")

    weight = []
    weight_each = 0
    for i in range(len(b)):
        for j in range(len(c)):
            #print(len(c), len(b))
            #print(c[j][0],b[j][i])
            weight_each += b[i][j]*c[j][0]
        weight.append(weight_each)
        weight_each = 0

import xlwt

#设置表格样式
def set_style(name,height,bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style

#写Excel
def write_excel():
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('Data parameters',cell_overwrite_ok=True)
    row0 = inputs_name
    colum0 = [""]
    #写第一行
    for i in range(0,len(row0)):
        sheet1.write(0,i,row0[i],set_style('Times New Roman',220,True))
    #写第一列
    for i in range(0,len(colum0)):
        sheet1.write(i+1,0,colum0[i],set_style('Times New Roman',220,True))

    for i in range(len(weight)-1):
        sheet1.write(1,i,weight[i])
    f.save('test.xls')

write_excel()
    #sheet1.write_merge(6,6,1,3,'未知')#合并行单元格
    #sheet1.write_merge(1,2,3,3,'打游戏')#合并列单元格
    #sheet1.write_merge(4,5,3,3,'打篮球')

  


eg.msgbox("Hey, Susu has the solution now!")

