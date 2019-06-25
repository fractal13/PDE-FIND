#!/usr/bin/env python3

import sys
sys.path.append( ".." )

import numpy as np
from matplotlib import pyplot as plt
from PDE_FIND import *
labelfontsize = 16; tickfontsize = 16; titlefontsize = 18

# Create the random walk data
length = 10**6
dt = 0.01
np.random.seed(0)
pos = np.cumsum(np.sqrt(dt)*np.random.randn(length))

    
# display data shape
plt.plot(dt*np.arange(length),pos, linewidth = 1.5)
plt.xlabel('Time', fontsize = labelfontsize)
plt.ylabel('Location', fontsize = labelfontsize)
plt.title('A single trace of Brownian motion', fontsize = titlefontsize)
plt.xticks(fontsize = tickfontsize); plt.yticks(fontsize = tickfontsize)

plt.savefig('rw.png')
plt.cla( ); plt.clf( );

# convert to histogram and plot
P = {}
M = 0

m = 5
n = 300

for i in range(m):
    P[i] = []
    
for i in range(len(pos)-m):
    
    # center
    y = pos[i+1:i+m+1] - pos[i]
    M = max([M, max(abs(y))])
    
    # add to distribution
    for j in range(m):
        P[j].append(y[j])
    
bins = np.linspace(-M,M,n+1)
x = np.linspace(M*(1/n-1),M*(1-1/n),n)
dx = x[2]-x[1]
T = np.linspace(0,dt*(m-1),m)
U = np.zeros((n,m))
for i in range(m):
    U[:,i] = plt.hist(P[i],bins,label=r'$t = $' + str(i*dt+dt))[0]/float(dx*(len(pos)-m))
    
plt.xlabel('Location', fontsize = labelfontsize)
plt.ylabel(r'$f(x,t)$', fontsize = labelfontsize)
plt.title(r'Histograms for $f(x,t)$', fontsize = titlefontsize)
plt.xticks(fontsize = tickfontsize); plt.yticks(fontsize = tickfontsize)
plt.legend(loc = 'upper right', fontsize = 14)

plt.savefig('rw-hist.png')

# Construct Theta( U ), and compute U_t
Ut, R, rhs_des = build_linear_system(U, dt, dx, D=4, P=5, time_diff = 'FD', deg_x = 4)

# Solve with STRidge
w = TrainSTRidge(R, Ut, 10**-2,10, normalize = 2)
print( "PDE derived using STRidge" )
print_pde(w, rhs_des)

sys.exit( 1 )

