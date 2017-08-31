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


def Re(kin_visc,c,U):
    return U*c/kin_visc

def cla(U,kin_visc,c):
    filename="naca0012_"+str(int(U))
    calcPolar("0012", int(Re(kin_visc,c,U)), filename, alfaseq=np.linspace(-5,5,10))
    inp = np.loadtxt(filename,skiprows=12)
    
    AOA=inp[:,0]*np.pi/180.
    CL=inp[:,1]
    p=polyfit(AOA,CL,1)
    return(p[0])

def Ka(k,x_ea,x_sp):
    return 4*k*np.abs(x_ea-x_sp)/2.
    
def div_crit(U,rho,l,c,x_ea,x_ac,x_sp,k,kin_visc): # divergence criteria
    S=l*c
    return Ka(k,x_ea,x_sp)-S*cla(U,kin_visc,c)*.5*U*U*rho*(x_ea-x_ac)

def get_div_speed(Umax,l,c,x_ea,x_sp,x_ac,k,plot=False,steps=20,Umin=.5,rho=1.2,kin_visc=15.11E-6):
    U=np.linspace(Umin,Umax,steps)
    crit=np.zeros(len(U))
    for i in range(len(U)):
        crit[i]=div_crit(U[i],rho,l,c,x_ea,x_ac,x_sp,k,kin_visc)
    if plot:
        fig, ax = plt.subplots()
        plt.plot(U,crit,".b")
        plt.plot(U[min(np.abs(crit))==np.abs(crit)],crit[min(np.abs(crit))==np.abs(crit)],'go')
        plt.xlabel(r'$\mathrm{Flow\ velocity\ [m/s]}$')
        plt.ylabel(r'$\mathrm{Effective\ stiffnesst\ [N/rad]}$')
        plt.title(r"$\mathrm{Divergence\ velocity\ from\ effective\ stiffness}$") 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')  
    return U[min(np.abs(crit))==np.abs(crit)]
    
def test_get_Ud():
    c=0.108 #m
    l=0.295 #m
    x_sp=0.6*c #m
    x_ea=0.5*c #m
    x_ac=0.25*c #m
    k=14.53#N/m    
    Umax=10 # m/s
    print(get_div_speed(Umax,l,c,x_ea,x_sp,x_ac,k,steps=40,plot=True))
    
test_get_Ud()    
#fmt = r'$%1.1f$'
#xticks = mtick.FormatStrFormatter(fmt)
#ax.xaxis.set_major_formatter(xticks)
#fmt = r'$%1.0f$'
#yticks = mtick.FormatStrFormatter(fmt)
#ax.yaxis.set_major_formatter(yticks)



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
