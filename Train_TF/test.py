#-*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from scipy.io import savemat


img=Image.open('test.png')
img=img.convert('L')
grey=img.getdata()
X=np.asarray(grey)
X=np.reshape(X,(1,784))
y=5

sess=tf.Session()
saver=tf.train.import_meta_graph("checkpoint\\Model.ckpt.meta")
saver.restore(sess,tf.train.latest_checkpoint('checkpoint'))
graph=tf.get_default_graph()
xs=graph.get_tensor_by_name('input:0')
ys=graph.get_tensor_by_name('output:0')
h=tf.get_collection('result')
ans=sess.run(h,feed_dict={xs:X})
print(np.argmax(ans),y)
print(ans)