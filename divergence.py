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
from scipy import interpolate

def generate_lookup_clalph(Umin,Umax,steps,l,c,x_ea,x_ac,x_sp,k,filename,rho=1.2,kin_visc=15.11E-6):
    U=np.linspace(Umin,Umax,steps)    
    crit=np.zeros(len(U))
    for i in range(len(U)):
        crit[i]=div_crit(U[i],rho,l,c,x_ea,x_ac,x_sp,k,kin_visc)
    np.save(filename, [U,crit]) 

def get_lookup_clalph(filename,plot=False):
    lookup=np.load(filename)
    U=lookup[0,:]
    cla=lookup[1,:]
    if plot:
        fig, ax = plt.subplots()
        u=np.linspace(Umin,Umax,steps)
        plt.plot(u,f(u),'or')
        plt.xlabel(r'$\mathrm{Flow\ velocity\ [m/s]}$')
        plt.ylabel(r'$\mathrm{c_{l,\alpha}}$')
        plt.title(r"$\mathrm{Divergence\ velocity\ from\ effective\ stiffness}$") 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left') 
        
        
        
    return interpolate.interp1d(U, cla)

def flutter_speed(filename,U,m,Ia,Sa,k,x_ea,x_sp,x_cg,x_ac,c,l,rho=1.2):   
    cla=get_lookup_clalph(filename)
    ka=Ka(k,x_ea,x_sp)    
    kh=4*k
    S=c*l
    q=.5*U*U*rho    
    e=x_ea-x_ac   
    
    A=m*Ia-Sa**2+0j
    B=kh*Ia+(ka-e*q*S*cla(U))*m-q*S*cla(U)*Sa+0j
    C=kh*(ka-e*q*S*cla(U))+0j
    p2_1=(-B+np.sqrt(B**2-4*A*C))/(2*A)    
    p2_2=(-B-np.sqrt(B**2-4*A*C))/(2*A) 
    print(A)
    print(B)
    print(C)
    print(p2_1)    
    print(p2_2)
    
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
    return 8*k/np.abs(x_ea-x_sp)
    
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

def test_get_flutter():

    Umin=0.5
    Umax=15
    steps=500
    
    c=0.108 #m
    l=0.295 #m
    spring_pos=np.array([0.2,0.5])*c
    x_sp=0.5*c #m
    x_ea=np.mean(spring_pos) #m
    x_ac=0.25*c #m
    x_cg=0.4545*c #m
    m=0.071 #kg
    Ia=3.92E-4 #kg/m/l
    Sa=(x_ea-x_cg)*m #kg m
    k=2*14.53#N/m  
    filename="cl_alpha_lookup.npy"
    U=9.
    flutter_speed(filename,U,m,Ia,Sa,k,x_ea,x_sp,x_cg,x_ac,c,l)
 
test_get_flutter()

#generate_lookup_clalph(Umin,Umax,steps,l,c,x_ea,x_ac,x_sp,k,filename,rho=1.2,kin_visc=15.11E-6)






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
