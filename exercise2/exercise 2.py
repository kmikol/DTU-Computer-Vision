# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:50:39 2021

@author: Youyang Shen
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math

def box3d(n):
    array = np.linspace(-0.5,0.5,n+2)
    array = array[1:-1]
    fig = plt.figure()
    array_0 = np.zeros(n)
    array_pos = np.ones(n)*0.5
    array_neg = np.ones(n)*(-0.5)
    M = np.array([array_0,array_0,array]).T
    M1 = np.array([array,array_0,array_0]).T
    M = np.vstack((M,M1))
    M1 = np.array([array_0,array,array_0]).T
    M = np.vstack((M,M1))
    M1 = np.array([array_pos,array,array_pos]).T
    M = np.vstack((M,M1))
    M1 = np.array([array_neg,array,array_pos]).T
    M = np.vstack((M,M1))
    M1 = np.array([array_pos,array,array_neg]).T
    M = np.vstack((M,M1))
    M1 = np.array([array_neg,array,array_neg]).T
    M = np.vstack((M,M1))    
    M1 = np.array([array,array_pos,array_pos]).T
    M = np.vstack((M,M1))
    M1 = np.array([array,array_neg,array_pos]).T
    M = np.vstack((M,M1))
    M1 = np.array([array,array_pos,array_neg]).T
    M = np.vstack((M,M1))
    M1 = np.array([array,array_neg,array_neg]).T
    M = np.vstack((M,M1))
    M1 = np.array([array_pos,array_pos,array]).T
    M = np.vstack((M,M1))
    M1 = np.array([array_pos,array_neg,array]).T
    M = np.vstack((M,M1))
    M1 = np.array([array_neg,array_pos,array]).T
    M = np.vstack((M,M1))
    M1 = np.array([array_neg,array_neg,array]).T
    M = np.vstack((M,M1))    
    array1 = [0.5,-0.5,0.5,0.5,-0.5,0.5,-0.5,-0.5,0,0,0,0,0.5,-0.5]
    array2 = [0.5,0.5,-0.5,0.5,-0.5,-0.5,0.5,-0.5,0,0,0.5,-0.5,0,0]
    array3 = [0.5,0.5,0.5,-0.5,0.5,-0.5,-0.5,-0.5,0.5,-0.5,0,0,0,0]
    M1 = np.array([array1,array2,array3]).T
    M = np.vstack((M,M1))  
    ax1 = plt.axes(projection='3d') 
    ax1.scatter3D(M[:,0],M[:,1],M[:,2], cmap='Blues')
    return M

def projectpoints(K,R,t,Q):
    Rt = np.hstack((R,t.T))
    p = K*Rt
    result = p*Q
    result = result/result[2]
    return result

K = [[800,0,100],[0,800,100],[0,0,1]]
R = np.identity(3)
t = np.matrix([0,-1,-5])
Q0 =np.matrix([1,2,15,1]).T

def projectpointss(K,R,t,dist,Q):
    Ka = [[1,0,K[0][2]],[0,1,K[1][2]],[0,0,1]]
    Kb = [[K[0][0],0,0],[0,K[0][0],0],[0,0,1]]
    k3=dist[0]
    k5=dist[1]
    k7=dist[2]
    Rt = np.hstack((R,t.T))
    p1 = Kb*Rt
    result1 = p1*Q
    result1 = result1/result1[2][0]
    r = math.sqrt(pow(result1[0][0],2)+pow(result1[1][0],2))
    delta_r = k3*pow(r,2)+k5*pow(r,4)+k7*pow(r,6)
    result2 = (1+delta_r)*result1
    result2[2][0]=1
    result2 = Ka*result2
    #result2 = result2.tolist()
    #x = [result2[0][0],result2[1][0],1]
    return result2

dist = [-pow(10,-5),0,0]

A = (projectpointss(K, R, t,dist, Q0))

def RR(x,y,z):
    Ma = np.array([[math.cos(z),-math.sin(z),0],[math.sin(z),math.cos(z),0],[0,0,1]])
    Mb = np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    Mc = np.array([[1,0,0],[0,math.cos(x),-math.sin(x)],[0,math.sin(x),math.cos(x)]])
    result = np.dot(Ma,Mb)
    result = np.dot(result,Mc)
    return result

Rg = RR(1,1,-0.5)
Tg = np.matrix([0,0,-7])
Kg = [[4000,0,960],[0,4000,540],[0,0,1]]
distg_true = [-pow(10,-6),pow(10,-12),0]
distg_false = [-pow(10,-6),0,0]
Q = box3d(20)
n,m = Q.shape
qtrue = np.zeros((n,m))
qfalse = np.zeros((n,m))
Qone = np.ones((n,1))
Q = np.hstack((Q,Qone))
for i in range(n):
    Q_temp = np.matrix(Q[i]).T
    B = projectpointss(Kg, Rg, Tg, distg_true, Q_temp)
    qtrue[i]=B.T

for i in range(n):
    Q_temp = np.matrix(Q[i]).T
    B2 = projectpointss(Kg, Rg, Tg, distg_false, Q_temp)
    qfalse[i]=B2.T

q_delta = qtrue-qfalse
Err = sum(sum(pow(q_delta,2)))
print(Err)
avg_err = Err/n
print(avg_err)


