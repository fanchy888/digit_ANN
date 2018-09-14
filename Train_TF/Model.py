#-*- coding:utf-8 -*-
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import random
import tensorflow as tf

#training set
train_set=loadmat('train')
X=train_set['X'].astype(np.float32)
y=train_set['y']
m=y.shape[0]
yy=np.mat(np.zeros((m,10)))
for i in range(m):
	n=y[i]%10
	yy[i,n]=1
y=yy.astype(np.float32)

#test set
test_set=loadmat('test')
x_test=test_set['X'].astype(np.float32)
y_test=test_set['y']
m1=y_test.shape[0]
yy=np.mat(np.zeros((m1,10)))
for i in range(m1):
	n=y_test[i]%10
	yy[i,n]=1
y_test=yy.astype(np.float32)


s1=X.shape[1]
s2=30
s3=30
s4=10
alpha=0.001

xs=tf.placeholder(tf.float32,[None,s1],name='input')
ys=tf.placeholder(tf.float32,[None,s4],name='output')	

theta={
	1:tf.Variable(tf.random_uniform([s1,s2],-0.1,0.1)),
	2:tf.Variable(tf.random_uniform([s2,s3],-0.1,0.1)),
	3:tf.Variable(tf.random_uniform([s3,s4],-0.1,0.1)),
}	
	
bias={
	1:tf.Variable(tf.ones([s2])),
	2:tf.Variable(tf.ones([s3])),
	3:tf.Variable(tf.ones([s4])),
}	


def connect_layer(input):
	layer1=tf.add(tf.matmul(input,theta[1]),bias[1])
	layer2=tf.add(tf.matmul(layer1,theta[2]),bias[2])
	layer3=tf.add(tf.matmul(layer2,theta[3]),bias[3])
	return layer3
	
out=connect_layer(xs)
h=tf.nn.softmax(out)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out,labels=ys))
train_step=tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost)



correct_predict=tf.equal(tf.argmax(ys,1),tf.argmax(h,1))
accuracy=tf.reduce_mean(tf.cast(correct_predict,tf.float32))

saver=tf.train.Saver()
init=tf.global_variables_initializer()

tf.add_to_collection('train_step',train_step)
tf.add_to_collection('J',cost)
tf.add_to_collection('acc',accuracy)
tf.add_to_collection('result',h)
with tf.Session() as sess:
	sess.run(init)
	for i in range(200):
		sess.run(train_step,feed_dict={xs:X,ys:y})
		if i%10==0:
			print('cost:',sess.run(cost,feed_dict={xs:X,ys:y}))
			print('acc: ',sess.run(accuracy,feed_dict={xs:x_test,ys:y_test}))


	save_path=saver.save(sess,'checkpoint\\Model.ckpt')
	
