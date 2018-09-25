#-*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from scipy.io import savemat

X=np.empty([10,784])
y=[]
for i in range(10):
	img=Image.open(str(i)+'.png')
	img=img.convert('L')
	grey=img.getdata()
	xi=np.asarray(grey)
	xi=np.reshape(xi,(1,784))
	X[i]=xi
	y.append(i)

sess=tf.Session()
saver=tf.train.import_meta_graph("checkpoint\\Model.ckpt.meta")
saver.restore(sess,tf.train.latest_checkpoint('checkpoint'))
graph=tf.get_default_graph()
xs=graph.get_tensor_by_name('input:0')
ys=graph.get_tensor_by_name('output:0')
h=tf.get_collection('result')[0]
ans=tf.argmax(h,1)
print(sess.run(ans,feed_dict={xs:X}))
print(sess.run(h,feed_dict={xs:X})[6])