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
from numpy import linalg as LA
from numpy.linalg import inv
from numpy.linalg import det



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
    
    calcPolar(profile, int(Re(kin_visc,c,U)), filename, alfaseq=np.linspace(0,4,10))
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
    
def claclm(U,kin_visc,c,plot_cl=False,plot_cm=False,profile="0012",pdfobj=-1): #1/rad 
    filename='naca'+profile+"_"+str(int(U))   
    calcPolar(profile, int(Re(kin_visc,c,U)), filename, alfaseq=np.linspace(0,4,10))
    inp = np.loadtxt(filename,skiprows=12)
    inp = np.append(inp[:,:],-inp[:,:],axis=0)

    AOA=inp[:,0]
    CL=inp[:,1]
    CM=inp[:,4]

    p1=polyfit(AOA*np.pi/180.,CL,1)
    p2=polyfit(AOA*np.pi/180.,CM,1)
    if plot_cl:
        fig, ax = plt.subplots()
        plt.plot(AOA*np.pi/180.,CL,".",label=r"$\mathrm{Num. data}$",color=[.2, .2, .2])  
        plt.plot(AOA*np.pi/180.,p1[0]*AOA*np.pi/180.,label=r"$\mathrm{Linear regression}$",color="red")
        plt.title(r"$\mathrm{NACA"+profile+"}\ U_\infty="+str('%0.2f' %U)+"\ ms^{-1}\ \mathrm{Re}="+str(int(Re(kin_visc,c,U)))+"$")
        plt.xlabel(r"$\mathrm{Angle\ of\ attack}\ [rad]$")
        plt.ylabel(r"$c_l\ [-]$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')  
        if pdfobj!=-1:
           pdfobj.savefig()   
    if plot_cm:
        fig, ax = plt.subplots()
        print(CM)
        print(AOA)
        plt.plot(AOA*np.pi/180.,CM,".",label=r"$\mathrm{Num. data}$",color=[.2, .2, .2])  
        plt.plot(AOA*np.pi/180.,p2[0]*AOA*np.pi/180.,label=r"$\mathrm{Linear regression}$",color="red")
        plt.title(r"$\mathrm{NACA"+profile+"}\ U_\infty="+str('%0.2f' %U)+"\ ms^{-1}\ \mathrm{Re}="+str(int(Re(kin_visc,c,U)))+"$")
        plt.xlabel(r"$\mathrm{Angle\ of\ attack}\ [rad]$")
        plt.ylabel(r"$c_m\ [-]$")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')  
        if pdfobj!=-1:
           pdfobj.savefig() 
           
    return(p1[0],p2[0])

def Ka(k,x_ea,x_sp):
    return k*np.abs(x_ea-x_sp)**2
    
def eff_stiff(U,rho,l,c,x_ea,x_ac,x_sp,k,kin_visc,profile="0012",pdfobj=-1): # effective stiffness
    S=l*c
    return 8*Ka(k,x_ea,x_sp)-S*cla(U,kin_visc,c,profile=profile,pdfobj=pdfobj)*.5*U*U*rho*(x_ea-x_ac)

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
        fmt = r'$%1.1f$'
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
    Umax=5. # m/s
    pp = PdfPages("naca0012_divergencespeed")
    get_div_speed(5,l,c,x_ea,x_sp,x_ac,k,steps=20,plot=True,profile="0012",pdfobj=pp)
    pp.close()
    
def get_flutter(plot=False):
    
    c=0.108
    x_ea=0.35*c # elastic axis [m]
    x_cg=0.4545*c # center of gravity [m]
    x_ac=0.25*c # aerodynamic center [m]
    x_sp=0.5*c # spring location
    
    k=14.53 #N/m
    kh=8*k #N/m
    ka=8*Ka(k,x_ea,x_sp) #Nm/rad
    
    m=0.071 #kg
    Ia=3.92E-4 #kg m
    
    l=0.295
    S=c*l #m**2
    
    Sa=(x_ea-x_cg)*m #kg m
    e=x_ea-x_ac #m
    
    Umin=0.5
    Umax=50
    
    U=np.linspace(Umin,Umax,1)
    EVs=np.zeros([len(U),3])+0j
    EVqs=np.zeros([len(U),5])+0j
    for i in range(len(U)):
#        EVs[i,:]=np.append(U[i],flutter_st(U[i],kh,ka,S,m,Sa,Ia,e,c))
        EVqs[i,:]=np.append(U[i],flutter_qst(U[i],kh,ka,S,m,Sa,Ia,e,c))
        
        
    if plot:
        fig, ax = plt.subplots()
        plt.plot(EVqs[:,0],np.imag(EVqs[:,1]),".",color=[.05,.05,.05],label=r"$\mathrm{Eigenvalue}\ 1$")
        plt.plot(EVqs[:,0],np.imag(-EVqs[:,1]),".",color=[.05,.05,.05])
        plt.plot(EVqs[:,0],np.imag(EVqs[:,2]),".",color=[.1,.5,.1],label=r"$\mathrm{Eigenvalue}\ 2$")
        plt.plot(EVqs[:,0],-np.imag(EVqs[:,2]),".",color=[.1,.5,.1])
        plt.legend(loc=3)
        plt.xlabel(r'$\mathrm{Flow\ velocity\ [m/s]}$')
        plt.ylabel(r'$\mathrm{Im}\ EV$')
        plt.title(r"$\mathrm{Imag.\ part\ of\ eigenvalues\ static}$") 
        fmt = r'$%1.0f$'
        xticks = mtick.FormatStrFormatter(fmt)
        ax.xaxis.set_major_formatter(xticks) 
        yticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)     
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')      

# static
def flutter_st(U,Kh,Ka,S,m,Sa,Ia,e,c,kin_visc=15.11E-6,profile="0012",rho=1.22):
    # generate matrix
    dclda,dcmda=cladcm(U,kin_visc,c,plot=False)
    q=.5*rho*U**2
    A=np.array([[Kh, q*S*dclda],[0, Ka-e*q*S*dclda]])
    B=inv(np.array([[m, Sa],[Sa, Ia]]))
    ev=np.asarray(np.sqrt(LA.eig(np.dot(B,A))[0]+0j))    
    return ev
    
# quasi-static
def flutter_qst(U,Kh,Ka,S,m,Sa,Ia,e,c,kin_visc=15.11E-6,profile="0012",rho=1.22):
    dclda,dcmda=claclm(U,kin_visc,c,plot_cm=False,plot_cl=False)
    q=.5*rho*U**2
    
    M=np.array([[m, Sa],[Sa, Ia]])
    D=np.array([[q*S*dclda/U,0],[e*q*S*dclda/U,-q*S*c*dcmda]])
    K=np.array([[Kh, q*S*dclda],[0, Ka-e*q*S*dclda]])
    
    A=np.zeros([4,4])
    A[:2,:2]=M
    A[2:,2:]=np.diag(np.ones(2))
    
    B=np.zeros([4,4])
    B[:2,:2]=D
    B[:2,2:]=K
    B[2:,:2]=-np.diag(np.ones(2))
    
    Ai=inv(A)
    ev=LA.eig(np.dot(Ai,B)+0j)  
    print(ev)
    return ev[0]
    
# theodorsen garrik    
def flutter_theodorsen_garrik():
    return 0   



#get_Ud()
get_flutter(plot=True)

#generate_lookup_clalph(Umin,Umax,steps,l,c,x_ea,x_ac,x_sp,k,filename,rho=1.2,kin_visc=15.11E-6)
#get_Ud()






