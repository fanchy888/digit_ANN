#-*- coding:utf-8 -*-
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import random
import tensorflow as tf

	
train_set=loadmat('train')
X=train_set['X'].astype(np.float32)
y=train_set['y']
m=y.shape[0]
yy=np.mat(np.zeros((m,10)))
for i in range(m):
	n=y[i]%10
	yy[i,n]=1
y=yy.astype(np.float32)


test_set=loadmat('test')
x_test=test_set['X'].astype(np.float32)
y_test=test_set['y']
m1=y_test.shape[0]
yy=np.mat(np.zeros((m1,10)))
for i in range(m1):
	n=y_test[i]%10
	yy[i,n]=1
y_test=yy.astype(np.float32)



sess=tf.Session()
saver=tf.train.import_meta_graph("checkpoint\\Model.ckpt.meta")
saver.restore(sess,tf.train.latest_checkpoint('checkpoint'))
graph=tf.get_default_graph()
xs=graph.get_tensor_by_name('input:0')
ys=graph.get_tensor_by_name('output:0')
train=tf.get_collection('train_step')
cost=tf.get_collection('J')
acc=tf.get_collection('acc')

for i in range(1000):
	sess.run(train,feed_dict={xs:X,ys:y})
	if i%20==0:
		print('Iter No:',i)
		print(' Cost:',sess.run(cost,feed_dict={xs:X,ys:y})[0])
		print(' Accuracy:',sess.run(acc,feed_dict={xs:x_test,ys:y_test})[0])
saver.save(sess,'checkpoint\\Model.ckpt')


