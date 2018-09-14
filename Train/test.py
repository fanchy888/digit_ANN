#-*- coding:utf-8 -*-
from PIL import Image
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
def sigmoid(z):
	g=1/(1+np.exp(-z))
	return g
img=Image.open('test.png')
img=img.convert('L')
grey=img.getdata()

X=np.asarray(grey)
X=np.mat(X.ravel())
theta=loadmat('theta')

theta1=theta['theta1']
theta2=theta['theta2']
theta3=theta['theta3']
a1=np.hstack((np.mat(np.ones((1,1))),X))
a2=sigmoid(a1*theta1.T)
a2=np.hstack((np.mat(np.ones((1,1))),a2))
a3=sigmoid(a2*theta2.T)
a3=np.hstack((np.mat(np.ones((1,1))),a3))
h=sigmoid(a3*theta3.T)

y1=np.argmax(h,axis=1)
print(h)
train_set=loadmat('test')
weight=loadmat('theta')
X=np.mat(train_set['X'])
y=np.mat(train_set['y'])
m=y.shape[0]

a1=np.hstack((np.mat(np.ones((m,1))),X))
a2=sigmoid(a1*theta1.T)
a2=np.hstack((np.mat(np.ones((m,1))),a2))
a3=sigmoid(a2*theta2.T)
a3=np.hstack((np.mat(np.ones((m,1))),a3))
h=sigmoid(a3*theta3.T)
y1=np.argmax(h,axis=1)
accuracy=np.mean(np.double(y1==y))*100
print(accuracy)


