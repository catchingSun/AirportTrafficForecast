import numpy as np
from scipy.misc import derivative
import matplotlib.pylab as plt


def identy(x):
    return x


def forward(xin, connt, ):

    x_d1,x_d2 = xin.shape
    c_d1,c_d2 = connt.shape
    assert x_d1+1==c_d1

    zin = np.vstack((xin,np.ones((1,x_d2))))
    a = np.array(np.mat(connt).T * np.mat(zin))
    z = actfc(a)

    return a,z


def backward(dtin,ain,zin,connt,actfc):

    d_d1,d_d2 = dtin.shape
    a_d1,a_d2 = ain.shape
    z_d1,z_d2 = zin.shape
    c_d1,c_d2 = connt.shape
    assert (d_d1==c_d2) & (a_d1==z_d1==c_d1-1)

    dt = np.array(np.mat(connt[0:c_d1-1,:]) * np.mat(dtin))
    dt = derivative(actfc,ain,order=5)*dt

    z = np.vstack((zin,np.ones((1,z_d2))))

    e = np.zeros((c_d1,c_d2))
    for i in range(0,c_d1):
        e[i] = np.sum(z[i]*dtin,axis=1)/d_d2
        ##print(e.shape)
    return e,dt


x = np.array([np.linspace(-7, 7, 200)])
t = np.cos(x) * 0.5

units = np.array([1,8,5,3,1])
units_bias = units+1
##print(units_bias)

connt = {}
for i in range(1,len(units_bias)):
    connt[i] = np.random.uniform(-1,1,size=(units_bias[i-1],units[i]))
##print(connt)

actfc = {0:identy}
for i in range(1,len(units_bias)-1):
    actfc[i] = np.tanh
actfc[len(units)-1] = identy
##print(actfc)

for k  in range(0,5000):
    a = {0:x}
    z = {0:actfc[0](a[0])}
    for i in range(1,len(units)):
        a[i],z[i] = forward(z[i-1],connt[i],actfc[i])

    dt = {len(units)-1:z[i]-t}
    e = {}
    for i in range(len(units)-1,0,-1):
        e[i],dt[i-1] = backward(dt[i],a[i-1],z[i-1],connt[i],np.tanh)

    pp = 0.05
    for i in range(1,len(units_bias)):
        connt[i] = connt[i]-pp*e[i]
        ##print(connt)

a = {0:x}
z = {0:actfc[0](a[0])}
for i in range(1,len(units)):
    a[i],z[i] = forward(z[i-1],connt[i],actfc[i])

plt.plot(x.T,t.T,'r',x,a[len(units)-1],'*')
plt.show()