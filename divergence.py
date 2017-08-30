# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 09:54:37 2017

Divergence Speed

@author: kniehaus


"""
import numpy as np
from scipy import polyfit
import matplotlib.pyplot as plt
from xfoil import calcPolar


kin_visc=15.11E-6 #m**2/s
c=0.108 #m
l=0.295 #m
m=0.13 #kg
x_sp=0.6*c #m
x_cg=0.35*c #m
x_ea=0.5*c #m
x_ac=0.25*c 
rho=1.2#kg/m**3
k=14.53#N/m

def Re(kin_visc,c,U):
    return U*c/kin_visc

def cl(U):
    filename="naca0012_"+str(int(U))
    calcPolar("0012", int(Re(kin_visc,c,U)), filename, alfaseq=np.linspace(-5,5,10))
    inp = np.loadtxt(filename,skiprows=12)
    
    AOA=inp[:,0]*np.pi/180.
    CL=inp[:,1]
    p=polyfit(AOA,CL,1)
    return(p[0])
def Ka(k,x_ea,x_sp):
    return 4*k*np.abs(x_ea-x_sp)/2.
    
def Ufc(U,rho,l,c,x_ea,x_sp,k): # u for divergence
    S=l*c
    return np.sqrt(2*Ka(k,x_ea,x_sp)/(rho*S*cl(U)*(x_ea-x_ac)))


U=np.linspace(.5,10,20)
Ud=np.zeros(len(U))
for i in range(len(U)):
    Ud[i]=Ufc(U[i],rho,l,c,x_ea,x_sp,k)

plt.plot(U,-U+Ud)

# make sure clalpha is the slope not cl
# make clalpha a function of u


#case="Remax"
#print(case)
#print(Ufc(case,rho,l,c,x_ea,x_sp,k))
#case="Remid"
#print(case)
#print(Ufc(case,rho,l,c,x_ea,x_sp,k))
#case="Remin"
#print(case)
#print(Ufc(case,rho,l,c,x_ea,x_sp,k))
#case="_4"
#print(case)
#print(Ufc(case,rho,l,c,x_ea,x_sp,k))
#case="_6"
#print(case)
#print(Ufc(case,rho,l,c,x_ea,x_sp,k))
