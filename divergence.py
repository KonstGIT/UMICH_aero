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
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages


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
    
def Re(kin_visc,c,U):
    return U*c/kin_visc

def cla(U,kin_visc,c,plot=True,profile="0012",pdfobj=-1): #1/rad
    filename='naca'+profile+"_"+str(int(U))
    
    calcPolar(profile, int(Re(kin_visc,c,U)), filename, alfaseq=np.linspace(0,8,50))
    inp = np.loadtxt(filename,skiprows=12)
    inp = np.append(inp,-inp[1:,:],axis=0)
    AOA=inp[:,0]
    CL=inp[:,1]
    p=polyfit(AOA*np.pi/180.,CL,1)
    if plot:
        fig, ax = plt.subplots()
        plt.plot(AOA*np.pi/180.,CL,".",label=r"$\mathrm{Num. data}$",color=[.2, .2, .2])  
        plt.plot(AOA*np.pi/180.,p[0]*AOA*np.pi/180.,label=r"$\mathrm{Linear regression}$",color="red")
        plt.title(r"$\mathrm{NACA"+profile+"}\ U_\infty="+str('%0.2f' %U)+"\ ms^{-1}\ \mathrm{Re}="+str(int(Re(kin_visc,c,U)))+"$")
        plt.xlabel(r"$\mathrm{Angle\ of\ attack}\ [rad]$")
        plt.ylabel(r"$c_l\ [-]$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')  
        if pdfobj!=-1:
           pdfobj.savefig()   
           
    return(p[0])

def Ka(k,x_ea,x_sp):
    return 4.*k*np.abs(x_ea-x_sp)
    
def eff_stiff(U,rho,l,c,x_ea,x_ac,x_sp,k,kin_visc,profile="0012",pdfobj=-1): # effective stiffness
    S=l*c
    return Ka(k,x_ea,x_sp)-S*cla(U,kin_visc,c,profile=profile,pdfobj=pdfobj)*.5*U*U*rho*(x_ea-x_ac)

def get_div_speed(Umax,l,c,x_ea,x_sp,x_ac,k,plot=False,steps=20,Umin=.5,rho=1.2,kin_visc=15.11E-6,profile="0012",pdfobj=-1):
    U=np.linspace(Umin,Umax,steps)
    crit=np.zeros(len(U))
    for i in range(len(U)):
        crit[i]=eff_stiff(U[i],rho,l,c,x_ea,x_ac,x_sp,k,kin_visc,profile,pdfobj=pdfobj)
        
    if plot:
        fig, ax = plt.subplots()
        plt.plot(U,crit,".b")
        plt.plot(U[min(np.abs(crit))==np.abs(crit)],crit[min(np.abs(crit))==np.abs(crit)],'go')
        plt.xlabel(r'$\mathrm{Flow\ velocity\ [m/s]}$')
        plt.ylabel(r'$\mathrm{Effective\ stiffness\ [N/rad]}$')
        plt.title(r"$\mathrm{Effective\ stiffness}$") 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')  
        plt.xlim([0,Umax*1.1])  
        fmt = r'$%1.0f$'
        xticks = mtick.FormatStrFormatter(fmt)
        ax.xaxis.set_major_formatter(xticks)            
        if pdfobj!=-1:
           pdfobj.savefig()  
           
    return U[min(np.abs(crit))==np.abs(crit)]
    
def get_Ud():
    c=0.108 #m
    l=0.295 #m
    x_sp=0.6*c #m
    x_ea=0.5*c #m
    x_ac=0.25*c #m
    k=14.53 #N/m    
    Umax=10. # m/s
    pp = PdfPages("naca0012_divergencespeed")
    get_div_speed(Umax,l,c,x_ea,x_sp,x_ac,k,steps=20,plot=True,profile="0012",pdfobj=pp)
    pp.close()
    
def test_get_flutter():

    Umin=2
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
    lookup_file="cl_alpha_lookup.npy"
    U=9.

    flutter_speed(lookup_file,U,m,Ia,Sa,k,x_ea,x_sp,x_cg,x_ac,c,l)

 
#test_get_Ud()

#generate_lookup_clalph(Umin,Umax,steps,l,c,x_ea,x_ac,x_sp,k,filename,rho=1.2,kin_visc=15.11E-6)
get_Ud()





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
