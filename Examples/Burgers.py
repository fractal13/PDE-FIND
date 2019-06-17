#!/usr/bin/env python3

import sys
sys.path.append( ".." )

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PDE_FIND import *
import scipy.io as sio
import itertools

# load data from matlab file
data = sio.loadmat('./burgers.mat')
u = np.real(data['usol'])
x = np.real(data['x'][0])
t = np.real(data['t'][:,0])
dt = t[1]-t[0]
dx = x[2]-x[1]


# display data shape
X, T = np.meshgrid(x, t)
fig1 = plt.figure()
ax = fig1.gca(projection='3d')
surf = ax.plot_surface(X, T, u.T, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
    linewidth=0, antialiased=False)
plt.title('Burgers Equation', fontsize = 20)
plt.xlabel('x', fontsize = 16)
plt.ylabel('t', fontsize = 16)
fig1.savefig('eq.png')

# Construct Theta( U ), and compute U_t
Ut, R, rhs_des = build_linear_system(u, dt, dx, D=3, P=3, time_diff = 'FD', space_diff = 'FD')
#print( ['1'] + rhs_des[1:] )
#print( Ut )
#print( R )


# Solve with STRidge
w = TrainSTRidge(R,Ut,10**-5,1)
print( "PDE derived using STRidge" )
print_pde(w, rhs_des)


# show the error
err = abs(np.array([(1 -  1.000987)*100, (.1 - 0.100220)*100/0.1]))
print( "Error using PDE-FIND to identify Burger's equation:\n" )
print( "Mean parameter error:", np.mean(err), '%' )
print( "Standard deviation of parameter error:", np.std(err), '%' )


####
####
#### Redo, after adding noise to the data
####
####
np.random.seed(0)
un = u + 0.01*np.std(u)*np.random.randn(u.shape[0],u.shape[1])


Utn, Rn, rhs_des = build_linear_system(un, dt, dx, D=3, P=3, time_diff = 'poly',
                                       deg_x = 4, deg_t = 4, 
                                       width_x = 10, width_t = 10)

# Solve with STRidge
w = TrainSTRidge(Rn,Utn,10**-5,1)
print( "PDE derived using STRidge" )
print_pde(w, rhs_des)

err = abs(np.array([(1 -  1.009655)*100, (.1 - 0.102966)*100/0.1]))
print( "Error using PDE-FIND to identify Burger's equation with added noise:\n" )
print( "Mean parameter error:", np.mean(err), '%' )
print( "Standard deviation of parameter error:", np.std(err), '%' )
