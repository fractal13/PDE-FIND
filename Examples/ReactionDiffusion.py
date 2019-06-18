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
data = sio.loadmat('./reaction_diffusion_big.mat')
t = data['t'][:,0]
x = data['x'][0,:]
y = data['y'][0,:]
U = data['u']
V = data['v']

n = len(x) # also the length of y
steps = len(t)
dx = x[2]-x[1]
dy = y[2]-y[1]
dt = t[2]-t[1]

# plot the data
plt.figure()
xx, yy = np.meshgrid(
    np.arange(n)*dx,
    np.arange(n)*dy)
plt.subplot(1,2,1)
plt.pcolor(xx,yy,U[:,:,10],cmap='coolwarm')
plt.title('U', fontsize = 20)
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.subplot(1,2,2)
plt.pcolor(xx,yy,V[:,:,10],cmap='coolwarm')
plt.title('V', fontsize = 20)
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.savefig('rd.png')


#
# sub-sample data (sparse data)
#
# every 6th time step is selected
# x,y coordinates are randomly selected
#
np.random.seed(0) # so that numbers in paper are reproducible
num_xy = 5000 # needs to be very high to work with noise
num_t = 30
num_points = num_xy * num_t
boundary = 5
points = {}
count = 0
for p in range(num_xy):
    x = np.random.choice(np.arange(boundary,n-boundary),1)[0]
    y = np.random.choice(np.arange(boundary,n-boundary),1)[0]
    for t in range(num_t):
        points[count] = [x,y,6*t+10]
        count = count + 1


#
# Construct Theta( U ), and compute U_t
#

# Take up to second order derivatives.
u = np.zeros((num_points,1))
v = np.zeros((num_points,1))
ut = np.zeros((num_points,1))
vt = np.zeros((num_points,1))
ux = np.zeros((num_points,1))
uy = np.zeros((num_points,1))
uxx = np.zeros((num_points,1))
uxy = np.zeros((num_points,1))
uyy = np.zeros((num_points,1))
vx = np.zeros((num_points,1))
vy = np.zeros((num_points,1))
vxx = np.zeros((num_points,1))
vxy = np.zeros((num_points,1))
vyy = np.zeros((num_points,1))

N = 2*boundary-1  # number of points to use in fitting
Nt = N
deg = 4 # degree of polynomial to use

for p in points.keys():
    [x,y,t] = points[p]
    if p % 1000 == 0:
        print( "Point: ", p, x, y, t )
    
    # value of function
    u[p] = U[x,y,t]
    v[p] = V[x,y,t]
    
    # time derivatives
    ut[p] = PolyDiffPoint(U[x,y,t-(Nt-1)//2:t+(Nt+1)//2], np.arange(Nt)*dt, deg, 1)[0]
    vt[p] = PolyDiffPoint(V[x,y,t-(Nt-1)//2:t+(Nt+1)//2], np.arange(Nt)*dt, deg, 1)[0]
    
    # spatial derivatives
    ux_diff    = PolyDiffPoint(U[ x-(N-1)//2 : x+(N+1)//2, y,                       t ], np.arange(N)*dx, deg, 2)
    uy_diff    = PolyDiffPoint(U[ x,                       y-(N-1)//2 : y+(N+1)//2, t ], np.arange(N)*dy, deg, 2)
    vx_diff    = PolyDiffPoint(V[ x-(N-1)//2 : x+(N+1)//2, y,                       t ], np.arange(N)*dx, deg, 2)
    vy_diff    = PolyDiffPoint(V[ x,                       y-(N-1)//2 : y+(N+1)//2, t ], np.arange(N)*dy, deg, 2)
    ux_diff_yp = PolyDiffPoint(U[ x-(N-1)//2 : x+(N+1)//2, y+1,                     t ], np.arange(N)*dx, deg, 2)
    ux_diff_ym = PolyDiffPoint(U[ x-(N-1)//2 : x+(N+1)//2, y-1,                     t ], np.arange(N)*dx, deg, 2)
    vx_diff_yp = PolyDiffPoint(V[ x-(N-1)//2 : x+(N+1)//2, y+1,                     t ], np.arange(N)*dx, deg, 2)
    vx_diff_ym = PolyDiffPoint(V[ x-(N-1)//2 : x+(N+1)//2, y-1,                     t ], np.arange(N)*dx, deg, 2)
    
    ux[p]  = ux_diff[0]
    uy[p]  = uy_diff[0]
    uxx[p] = ux_diff[1]
    uxy[p] = (ux_diff_yp[0]-ux_diff_ym[0])/(2*dy)
    uyy[p] = uy_diff[1]
    
    vx[p]  = vx_diff[0]
    vy[p]  = vy_diff[0]
    vxx[p] = vx_diff[1]
    vxy[p] = (vx_diff_yp[0]-vx_diff_ym[0])/(2*dy)
    vyy[p] = vy_diff[1]


print( "Points finished " )
# Form a huge matrix using up to quadratic polynomials in all variables.
X_data = np.hstack([u,v])
X_ders = np.hstack([np.ones((num_points,1)), ux, uy, uxx, uxy, uyy, vx, vy, vxx, vxy, vyy])
X_ders_descr = ['','u_{x}', 'u_{y}','u_{xx}','u_{xy}','u_{yy}','v_{x}', 'v_{y}','v_{xx}','v_{xy}','v_{yy}']
X, description = build_Theta(X_data, X_ders, X_ders_descr, 3, data_description = ['u','v'])
print( ['1'] + description[1:] )


#
# Solve system now
#

# u_t's equation
c = TrainSTRidge(X,ut,10**-5,1)
print_pde(c, description)


# v_t's equation
c = TrainSTRidge(X,vt,10**-5,1)
print_pde(c, description, ut = 'v_t')

# and, the errors
err = abs(np.array([(0.1-0.099977)*100/0.1,  (0.1-0.100033)*100/0.1,
                    (0.1-0.100009)*100/0.1,  (0.1-0.099971)*100/0.1,
                    (1-0.999887)*100,        (1-1.000335)*100,
                    (1-0.999906)*100,        (1-0.999970)*100,
                    (1-0.999980)*100,        (1-0.999978)*100,
                    (1-0.999976)*100,        (1-1.000353)*100,
                    (1-0.999923)*100,        (1-1.000332)*100]))
print( np.mean(err) )
print( np.std(err) )

