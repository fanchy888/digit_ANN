#-*- coding:utf-8 -*-
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import random


def sigmoid(z):
	g=1/(1+np.exp(-z))
	return g
	
	
l=0.1# lambda

train_set=loadmat('train')
weight=loadmat('theta')
X=np.mat(train_set['X'])
y=np.mat(train_set['y'])


m=y.shape[0]
print(m)
s1=X.shape[1]
s2=30
s3=30
s4=10
yy=np.mat(np.zeros((m,10)))
for i in range(m):
	n=y[i]%10
	yy[i,n]=1
y=yy
#cost function

#theta1=np.mat(np.random.rand(s2,s1+1))*0.18-0.09
#theta2=np.mat(np.random.rand(s3,s2+1))*0.18-0.09
#theta3=np.mat(np.random.rand(s4,s3+1))*0.18-0.09
theta1=np.mat(weight['theta1'])
theta2=np.mat(weight['theta2'])
theta3=np.mat(weight['theta3'])
a1=0
a2=0
a3=0
J=0
r=0

def cost():
	global a1,a2,a3,h,theta1,theta2,theta3,J,r
	a1=np.hstack((np.mat(np.ones((m,1))),X))
	a2=sigmoid(a1*theta1.T)
	a2=np.hstack((np.mat(np.ones((m,1))),a2))
	a3=sigmoid(a2*theta2.T)
	a3=np.hstack((np.mat(np.ones((m,1))),a3))
	h=sigmoid(a3*theta3.T)
	J=-np.multiply((1-y),np.log(1-h))-np.multiply(y,np.log(h))
	J=J.sum()/m

	#regularization

	r1=np.multiply(theta1[:,1:],theta1[:,1:])
	r2=np.multiply(theta2[:,1:],theta2[:,1:])
	r3=np.multiply(theta3[:,1:],theta3[:,1:])
	r=(r1.sum()+r2.sum()+r3.sum())*l/m/2
	J=J+r
	return 
cost()
#back propagation
theta1_grad=np.mat(np.zeros(theta1.shape))
theta2_grad=np.mat(np.zeros(theta2.shape))
theta3_grad=np.mat(np.zeros(theta3.shape))


def back_propagation():
	global theta1_grad,theta2_grad,theta3_grad,theta1,theta2,theta3,a1,a2,a3,h
	cost()
	for i in range(m):
		delta4=(h[i]-y[i]).T
		sgrad=np.multiply(a3[i],1-a3[i])
		delta3=np.multiply(theta3.T*delta4,sgrad.T)	
		delta3=delta3[1:]		
		sgrad=np.multiply(a2[i],1-a2[i])
		delta2=np.multiply(theta2.T*delta3,sgrad.T)		
		delta2=delta2[1:]
		theta1_grad=theta1_grad+delta2*a1[i]
		theta2_grad=theta2_grad+delta3*a2[i]
		theta3_grad=theta3_grad+delta4*a3[i]
		
	theta1_grad=theta1_grad/m+l/m*theta1
	theta2_grad=theta2_grad/m+l/m*theta2
	theta3_grad=theta3_grad/m+l/m*theta3
	theta1_grad[:,1]=theta1_grad[:,1]-l/m*theta1[:,1]
	theta2_grad[:,1]=theta2_grad[:,1]-l/m*theta2[:,1]
	theta3_grad[:,1]=theta3_grad[:,1]-l/m*theta3[:,1]

	return
back_propagation()
g1=theta1_grad.ravel()
g2=theta2_grad.ravel()
g3=theta3_grad.ravel()
grad=np.hstack((g1,g2))
grad=np.hstack((grad,g3))

#gradient check
'''
grad1=np.zeros(theta1_grad.shape)
grad2=np.zeros(theta2_grad.shape)
grad3=np.zeros(theta3_grad.shape)
for i in range(theta1.shape[0]):
	for j in range(theta1.shape[1]):
		perturb=np.mat(np.zeros(theta1_grad.shape))
		perturb[i,j]=10**(-5)		
		a2=sigmoid(a1*(theta1+perturb).T)
		a2=np.hstack((np.mat(np.ones((m,1))),a2))
		a3=sigmoid(a2*theta2.T)
		a3=np.hstack((np.mat(np.ones((m,1))),a3))
		h=sigmoid(a3*theta3.T)
		J1=-np.multiply((1-y),np.log(1-h))-np.multiply(y,np.log(h))
		J1=J1.sum()/m

		a2=sigmoid(a1*(theta1-perturb).T)
		a2=np.hstack((np.mat(np.ones((m,1))),a2))
		a3=sigmoid(a2*theta2.T)
		a3=np.hstack((np.mat(np.ones((m,1))),a3))
		h=sigmoid(a3*theta3.T)
		J2=-np.multiply((1-y),np.log(1-h))-np.multiply(y,np.log(h))
		J2=J2.sum()/m
		grad1[i,j]=(J1-J2)/2/(10**(-5))
	print(np.sum(grad1[i,:]-theta1_grad[i,:]))	
for i in range(theta2.shape[0]):
	for j in range(theta2.shape[1]):
		perturb=np.zeros(theta2_grad.shape)
		perturb[i,j]=10**(-5)	
		a2=sigmoid(a1*theta1.T)
		a2=np.hstack((np.mat(np.ones((m,1))),a2))
		a3=sigmoid(a2*(theta2+perturb).T)
		a3=np.hstack((np.mat(np.ones((m,1))),a3))
		h=sigmoid(a3*theta3.T)
		J1=-np.multiply((1-y),np.log(1-h))-np.multiply(y,np.log(h))
		J1=J1.sum()/m
		
		a2=sigmoid(a1*theta1.T)
		a2=np.hstack((np.mat(np.ones((m,1))),a2))
		a3=sigmoid(a2*(theta2-perturb).T)
		a3=np.hstack((np.mat(np.ones((m,1))),a3))
		h=sigmoid(a3*theta3.T)
		J2=-np.multiply((1-y),np.log(1-h))-np.multiply(y,np.log(h))
		J2=J2.sum()/m
		grad2[i,j]=(J1-J2)/2/(10**(-5))
	print(np.sum(grad2[i,:]-theta2_grad[i,:]))
for i in range(theta3.shape[0]):
	for j in range(theta3.shape[1]):
		perturb=np.zeros(theta3_grad.shape)
		perturb[i,j]=10**(-5)
		a2=sigmoid(a1*theta1.T)
		a2=np.hstack((np.mat(np.ones((m,1))),a2))
		a3=sigmoid(a2*theta2.T)
		a3=np.hstack((np.mat(np.ones((m,1))),a3))
		h=sigmoid(a3*(theta3+perturb).T)
		J1=-np.multiply((1-y),np.log(1-h))-np.multiply(y,np.log(h))
		J1=J1.sum()/m
		
		h=sigmoid(a3*(theta3-perturb).T)
		J2=-np.multiply((1-y),np.log(1-h))-np.multiply(y,np.log(h))
		J2=J2.sum()/m
		grad3[i,j]=(J1-J2)/2/(10**(-5))
	print(np.sum(grad3[i,:]-theta3_grad[i,:]))
grad_check=np.hstack((grad1.ravel(),grad2.ravel()))
grad_check=np.hstack((grad_check,grad3.ravel()))
print((np.linalg.norm(grad)-np.linalg.norm(grad_check))/(np.linalg.norm(grad)+np.linalg.norm(grad_check)))
print(np.linalg.norm(grad), np.linalg.norm(grad_check))

'''
a=0.003
#gradient decent
def gradient_decent(alpha):
	global theta1,theta2,theta3,theta1_grad,theta2_grad,theta3_grad	
	theta1=theta1-theta1_grad*alpha
	theta2=theta2-theta2_grad*alpha
	theta3=theta3-theta3_grad*alpha
	back_propagation()
	a=random.randint(0,50000)
	print('  Cost: %.6f' %J)
	print('  test case:  y:' ,np.argmax(y[a,:]),' h:',np.argmax(h[a,:]))
	return
BeforeJ=J

for i in range(50):
	print('Training NO: ',i+1)
	decrease=J
	for j in range(20):		
		print('**IterNo:%d **' %j)
		gradient_decent(a)		
	improve=(decrease-J)/decrease*100
	print('After 20 iterations: cost decrease %.6f' % improve)
	if improve<=0.01:
		a=a/3
		print('Changing alpha to %.5f',a)




accuracy=np.mean(np.double(np.argmax(h,axis=1)==np.argmax(y,axis=1)))*100
savemat('theta',{'theta1':theta1,'theta2':theta2,'theta3':theta3})
print('Cost %.6f  DeltaJ: %.4e  %.2f' %(J,BeforeJ-J,accuracy))	



