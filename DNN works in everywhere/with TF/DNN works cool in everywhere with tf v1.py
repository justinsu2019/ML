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



#store the whole nn 1/3
saver = tf.train.Saver()


### start nn start ###

#with with sentence, we don't have to close session as it will be closed automatically
with tf.Session() as sess:

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

                prediction_value = sess.run(prediction, feed_dict={xs: data})

                #print out, \n is for enter, and [0][0] is to select the very first value in whole matrix
                print("accuracy is {}".format(round(1-sess.run(loss, feed_dict={xs:data, ys: label}),6)))

                # because "xxx" is str, so if we want '+' instead of ',' (because use , there will be a enter
                # so we will need to str(the value)
                #below is for predict only 1 line:
                print(i," times, Aim is {}".format(str(sess.run(prediction, feed_dict={xs: test})[0][0]).strip()))

                
    #store the whole nn 3/3

    (saveOrNot==1 and print("Save to path:", saver.save(sess,saver_file)) or 0)

### start nn end ###




### weights reader start ###
#首先，使用tensorflow自带的python打包库读取模型

model_save_path = 'r"Networksaver/'+str(inputFile)+'weights.ckpt"'

#model_reader = pywrap_tensorflow.NewCheckpointReader(r"Networksaver/EURDweights.ckpt")

model_reader = pywrap_tensorflow.NewCheckpointReader(model_save_path)
#然后，使reader变换成类似于dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()
#最后，循环打印输出

for key in var_dict:
    print("variable name: ", key)
    print(model_reader.get_tensor(key))

    #print out whole picture:
    #pprint.pprint(model_reader.get_variable_to_shape_map())


### high lights is here!!! start ###


print(model_reader.get_tensor("Layer/Weights/Variable")[0])
print(model_reader.get_tensor("Layer/Weights/Variable")[0][0])


print("1   ")
pprint.pprint(model_reader.get_tensor("Layer/Weights/Variable"))
print("2   ")
pprint.pprint(model_reader.get_tensor("Layer/Weights/Variable/Adam_1"))
print("3   ")
pprint.pprint(model_reader.get_tensor("Layer/Weights/Variable/Adam"))

print("4    ")
pprint.pprint(model_reader.get_tensor("Layer_1/Weights/Variable"))
print("5    ")
pprint.pprint(model_reader.get_tensor("Layer_1/Weights/Variable/Adam"))
print("6    ")
pprint.pprint(model_reader.get_tensor("Layer_1/Weights/Variable/Adam_1"))

### high lights is here!!! end ###  

### weights reader end ###


import heapq
x=0
y=0
sigmoid = x/(1+y)
plt.figure(figsize=(16,10))
plt.xlim([0,40])
plt.ylim([0,12])
b = model_reader.get_tensor("Layer/Weights/Variable")
c = model_reader.get_tensor("Layer_1/Weights/Variable")
e = 0

# colors
f = ['antiquewhite','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond','blue','blueviolet',
     'brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','cornsilk','crimson','cyan',
     'darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen',
     'darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue','darkslategray','darkturquoise',
     'darkviolet','deeppink','deepskyblue','dimgray','dodgerblue','firebrick','floralwhite','forestgreen','fuchsia',
     'gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow','honeydew','hotpink','indianred','indigo',
     'ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan',
     'lightgoldenrodyellow','lightgreen','lightgray','lightpink','lightsalmon','lightseagreen','lightskyblue',
     'lightslategray','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon','mediumaquamarine',
     'mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise',
     'mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive',
     'olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip',
     'peachpuff','peru','pink','plum','powderblue','purple','red','rosybrown','royalblue','saddlebrown','salmon',
     'sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','snow','springgreen',
     'steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow',
     'yellowgreen'
     ]


input_list = ['start','high', 'low',  'close', '5','40','60','200','13','MACD1','MACD SIGNAL','MACD HISTOGRAM']


plt.scatter(16,10,s=100,c='r')
plt.annotate('Output', xy=(16,10), xytext=(17,11.5),
        arrowprops=dict(facecolor='black', shrink=0.05),
        )

#1st input layer 
for j in range(len(b)):
    plt.scatter(j*2+5,1,s=50)
    plt.annotate(input_list[j], xy=(j*2+5,1), xytext=(j*2+4,1-0.8),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
    e += 0.1
    
    for k in range(len(b[0])):
        max_weight = sorted(b[j])[len(b[j])-1]     
        plt.scatter(k,5,s=50)
        plt.plot([k,16],[5,10],c=f[k],lw=b[j][k],alpha=max(b[j][k],0.001)/max_weight)
        

        plt.plot([j*2+5,k],[1,5],c=f[k],lw=b[j][k],alpha=max(b[j][k],0.001)/max_weight)

        if b[j][k]>sorted(b[j])[len(b[j])-3]:  #only print the top 3 weights
            plt.text((j*2+5+k)/2,(1+5)/3+e,r'$'+str('%.2f' % (b[j][k]))+'$',color=f[k])   #,fontsize=d[k]*40
        else:
            pass
    
plt.show()


### print out the name and value of variables ###




eg.msgbox("Hey, Susu has the solution now!")
