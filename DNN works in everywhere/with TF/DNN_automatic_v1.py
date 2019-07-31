
# what's new
'''
1) pack the DNN into a function, even giving so much parameters...lol
2) consider time difference to inputs
3) record result into files
'''


# need to update:
#1) add forecase result in


#Try to compare the difference bewteen every activation function


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




def DNN(target, saveOrNot, loadOrNot, inputFile, activation_function,label,data,lr,Biases,time_stamp,saveToFile):

    inputs_name = pd.read_excel(inputFile+".xlsx" ).columns
    test_data = []
    label = label[time_stamp:]
    data = data[:len(label)]
 

    
    for i in range(len(label), len(label)+1+time_stamp):
        test = np.array(ee.ee(inputFile,0).ds()[i])[np.newaxis,:] # test data is the last line normally, of course we can change it if it's not.
        test_data.append(test)
    print("test data is {}, lr = {}, Biases = {}".format(test_data, lr, Biases))
    saver_file = "Networksaver/"+str(inputFile)+str(time_stamp)+"weights.ckpt" # save file to this location
    #when lr more than 0.05, it may goes to be bad
    #best biases is 0.5 now, accuracy is 0.439767
    data_size = len(ee.ee(inputFile,0).ds()[0])
    hidden_size = 2*data_size
    label_size = len(ee.ee(inputFile,1).ds()[0])
    loop = 200000


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
                if i % 500 == 0:

                    prediction_value = sess.run(prediction, feed_dict={xs: data})

                    #print out, \n is for enter, and [0][0] is to select the very first value in whole matrix
                    print("loss is {}".format(round(sess.run(loss, feed_dict={xs:data, ys: label}),6)))

                    # because "xxx" is str, so if we want '+' instead of ',' (because use , there will be a enter
                    # so we will need to str(the value)
                    #below is for predict only 1 line:

                    result_set = []

                    for j in range(len(test_data)):
                        result=str(sess.run(prediction, feed_dict={xs: test_data[j]})[0][0]).strip()
                        result_set.append(result)
                        print(" {} times, and {} tests, Result is {}".format(i,j,result))

        #store the whole nn 3/3

        (saveOrNot==1 and print("Save to path:", saver.save(sess,saver_file)) or 0)

        model_save_path = 'Networksaver/'+str(inputFile)+str(time_stamp)+'weights.ckpt'
        #model_reader = pywrap_tensorflow.NewCheckpointReader(model_save_path)
        model_reader = pywrap_tensorflow.NewCheckpointReader(model_save_path)


    if saveToFile == 1:
            
        # record the final result
        b = model_reader.get_tensor("Layer/Weights/Variable")
        c = model_reader.get_tensor("Layer_1/Weights/Variable")

        d = []
        e = []
        sum_data = 0

        for k in range(len(label)):
            for i in range(len(b)):
                sum_data = 0
                for j in range(len(c)):
                    #print(len(b),len(c),data[0][i],b[i][j],c[j][0],label[0])
                    sum_data += data[k][i]*b[i][j]*c[j][0]
                #print(sum_data)
                if label[k][0] is not 0:
                    e.append(sum_data/label[k][0])
            d.append(e)
            e=[]


        dataFrame = pd.DataFrame(d)
        dataFrame.columns = inputs_name
        dataFrame.loc['coefficient'] = dataFrame.apply(lambda x: x.sum())
        dataFrame = dataFrame.T.sort_values('coefficient', ascending=False) #转置并排序
        #dataFrame = dataFrame.sort_values('weight_each_total_list', ascending= False)
        dataFrameResult = pd.DataFrame(result_set)
        dataFrame = pd.concat([dataFrame,dataFrameResult],axis=0,join='outer') # outer 是并集


        #dataFrame = dataFrame.sort_values('weight_each_total_list', ascending= False)
        
        dataFrame.to_excel(str(time_stamp)+"Analyzs_result.xlsx",sheet_name='weights_recording')

    else:
        pass
        



if __name__ == "__main__":
    ### controller start ###
    #123 stands for final, highest, lowest
    target = 1 # select which sheetname is our targeted label, refer to ee; tips: 1 stands for sheet 2
    saveOrNot = 1 # save the result parameters or not
    loadOrNot = 1 # load pre-saved parameters or not
    saveToFile = 1 # save the final result to files
    inputFile = 'Data_reading' # data source 
    activation_function = "elu" # tf activation function
    lr = 0.005 # learn rate
    Biases = 0.01 # created biases

    # to check if time matters, add both 2 parameters below.
    i = 3 # because we have i as an input required in DNN, so if we don't have to test time, we will need to input i into DNN to, so we declare it here.  Also we can do one-time time test by updating i.
    test_time = 0 # if we need to check time, 

    test_parameters = 0 # check which learn rate & biases works best
    label = ((target == 1 and ee.ee(inputFile,1).ds()) or (target == 2 and ee.ee(inputFile,2).ds()) or (target == 3 and ee.ee(inputFile,3).ds())) # if we need to change label, we change in excel sheet 1/2/3 and update it here
    data = ee.ee(inputFile,0).ds()[:len(label)] # input data source is here.


    if test_parameters == 1:
        # do loop to check what parameters works best
        for lr in [0.005, 0.05,0.1]:
            for Biases in [0.01, 0.05, 0.1, 0.5]:
                DNN(target, saveOrNot, loadOrNot, inputFile, activation_function, label, data,lr,Biases,i,saveToFile)
    elif test_time == 1:
        # do loop to check if there is any relationship between time and the result
        for i in [1,2,3,4]:
            label = ((target == 1 and ee.ee(inputFile,1).ds()[i:]) or (target == 2 and ee.ee(inputFile,2).ds()[i:]) or (target == 3 and ee.ee(inputFile,3).ds()[i:])) # if we need to change label, we change here
            data = ee.ee(inputFile,0).ds()[:len(label)]
            DNN(target, saveOrNot, loadOrNot, inputFile, activation_function, label, data,lr,Biases,i,saveToFile,)
    else:
        DNN(target, saveOrNot, loadOrNot, inputFile, activation_function, label, data,lr,Biases,i,saveToFile,)

    eg.msgbox("Hey, Sisi has the solution now!")
