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
from scipy.interpolate import interp1d
from scipy.linalg import eig
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
    
    Umin=2
    Umax=6
    
    U=np.linspace(Umin,Umax,100)
    EVs=np.zeros([len(U),5])+0j
    EVqs=np.zeros([len(U),5])+0j
    for i in range(len(U)):
        EVs[i,:]=np.append(U[i],flutter_st(U[i],kh,ka,S,m,Sa,Ia,e,c))
        EVqs[i,:]=np.append(U[i],flutter_qst(U[i],kh,ka,S,m,Sa,Ia,e,c))
#        print(U[i])
#        print(EVqs[i,:])
        
    if plot:
        fig, ax = plt.subplots(2,1)

#        plt.plot(EVqs[:,0],np.imag(EVqs[:,1]),".",color="red",label=r"$\mathrm{Imag}\ 1$")
#        plt.plot(EVqs[:,0],np.imag(EVqs[:,2]),".",color="red")
#        plt.plot(EVqs[:,0],np.imag(EVqs[:,3]),".",color="red")
#        plt.plot(EVqs[:,0],np.imag(EVqs[:,4]),".",color="red")   
        f=interp1d(EVqs[:,0],np.real(EVqs[:,4]))
        cross=-1
#        for i in range(len(U)-1):
#            if f(U[i+1])*f(U[i])<0:
#                cross=.5*U[i]**2*1.22
#                ax[0].plot(.5*U[i]**2*1.22,0,'or')     

        ax[0].plot(.5*EVqs[:,0]**2*1.22,np.real(EVqs[:,1]),"-",color="red",linestyle="dashed",label=r"$\mathrm{Re}(\lambda_1^{qst.})$")
        ax[0].plot(.5*EVqs[:,0]**2*1.22,np.real(EVqs[:,2]),"-",color="blue",label=r"$\mathrm{Re}(\lambda_2^{qst.})$")
        ax[0].plot(.5*EVqs[:,0]**2*1.22,np.real(EVqs[:,3]),"-",color="orange",label=r"$\mathrm{Re}(\lambda_3^{qst.})$")
        ax[0].plot(.5*EVqs[:,0]**2*1.22,np.real(EVqs[:,4]),"-",color="green",label=r"$\mathrm{Re}(\lambda_4^{qst.})$")   
        ax[0].plot(.5*EVs[:,0]**2*1.22,np.real(EVs[:,1]),"-",color="black",linestyle="dashed",label=r"$\mathrm{Re}(\lambda_1^{st.})$")
        ax[0].plot(.5*EVs[:,0]**2*1.22,np.real(EVs[:,2]),"-",color="grey",linestyle="dashed",label=r"$\mathrm{Re}(\lambda_2^{st.})$")               
        ax[0].plot(.5*EVs[:,0]**2*1.22,np.real(EVs[:,3]),"-",color="black",linestyle="dashed",label=r"$\mathrm{Re}(\lambda_1^{st.})$")
        ax[0].plot(.5*EVs[:,0]**2*1.22,np.real(EVs[:,4]),"-",color="grey",linestyle="dashed",label=r"$\mathrm{Re}(\lambda_2^{st.})$")               

        ax[0].legend(loc=3)
        ax[0].set_xlabel(r'$\mathrm{Dynamic\ pressure\ [kg\ m^{-1}\ s^{-2}]}$')
        ax[0].set_ylabel(r'$\xi\ \ [1/s]$')
        ax[0].set_title(r"$\mathrm{Damping}\lambda_j=\xi_j+i\omega_j$") 
        fmt = r'$%1.2f$'
        xticks = mtick.FormatStrFormatter(fmt)
        ax[0].xaxis.set_major_formatter(xticks) 
        yticks = mtick.FormatStrFormatter(fmt)
        ax[0].yaxis.set_major_formatter(yticks)     
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].yaxis.set_ticks_position('left')     
        ax[0].plot(cross*np.ones(2),[np.max(np.real(EVqs[:,4])),np.min(np.real(EVqs[:,3]))],'-r')   

        ax[1].plot(.5*EVqs[:,0]**2*1.22,np.imag(EVqs[:,1]),"-",color="red",linestyle="dashed",label=r"$\mathrm{Im}(\lambda_1)$")
        ax[1].plot(.5*EVqs[:,0]**2*1.22,np.imag(EVqs[:,2]),"-",color="blue",label=r"$\mathrm{Im}(\lambda_2)$")
        ax[1].plot(.5*EVqs[:,0]**2*1.22,np.imag(EVqs[:,3]),"-",color="orange",label=r"$\mathrm{Im}(\lambda_3)$")
        ax[1].plot(.5*EVqs[:,0]**2*1.22,np.imag(EVqs[:,4]),"-",color="green",label=r"$\mathrm{Im}(\lambda_4)$") 

        ax[1].plot(.5*EVs[:,0]**2*1.22,np.imag(EVs[:,1]),"-",color="black",linestyle="dashed",label=r"$\mathrm{Im}(\lambda_1^{st.})$")
        ax[1].plot(.5*EVs[:,0]**2*1.22,np.imag(EVs[:,2]),"-",color="grey",linestyle="dashed",label=r"$\mathrm{Im}(\lambda_2^{st.})$")               
        ax[1].plot(.5*EVs[:,0]**2*1.22,np.imag(EVs[:,3]),"-",color="black",linestyle="dashed",label=r"$\mathrm{Im}(\lambda_1^{st.})$")
        ax[1].plot(.5*EVs[:,0]**2*1.22,np.imag(EVs[:,4]),"-",color="grey",linestyle="dashed",label=r"$\mathrm{Im}(\lambda_2^{st.})$")               
        
        ax[1].legend(loc=3)
        ax[1].set_xlabel(r'$\mathrm{Dynamic\ pressure\ [kg\ m^{-1}\ s^{-2}]}$')
        ax[1].set_ylabel(r'$\omega\ \ [1/s]$')
        ax[1].set_title(r"$\mathrm{Frequencies}\lambda_j=\xi_j+i\omega_j$") 
        fmt = r'$%1.2f$'
        xticks = mtick.FormatStrFormatter(fmt)
        ax[1].xaxis.set_major_formatter(xticks) 
        yticks = mtick.FormatStrFormatter(fmt)
        ax[1].yaxis.set_major_formatter(yticks)     
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].yaxis.set_ticks_position('left')     
        ax[1].plot(cross*np.ones(2),[np.min(np.imag(EVqs[:,4])),np.max(np.imag(EVqs[:,3]))],'-r')   

# static
def flutter_st(U,Kh,Ka,S,m,Sa,Ia,e,c,kin_visc=15.11E-6,profile="0012",rho=1.22):
    # generate matrix
#    dclda=cla(U,kin_visc,c,plot=False)
    dclda=6
    q=.5*rho*U**2
    A=np.array([[Kh, q*S*dclda],[0, Ka-e*q*S*dclda]])
    B=np.array([[m, Sa],[Sa, Ia]])
    ev=eig(np.dot(A,-B))[0]
    print(np.sqrt(ev))
    return np.append(np.sqrt(ev),-np.sqrt(ev))
    
# quasi-static
def flutter_qst(U,Kh,Ka,S,m,Sa,Ia,e,c,kin_visc=15.11E-6,profile="0012",rho=1.22):
    #dclda=claclm(U,kin_visc,c,plot_cm=False,plot_cl=False)
    dclda=6.1
    dcmdadot=-0.0
    q=.5*rho*U**2
    
#    M=np.array([[1,-1],[1,1]])+.0
#    D=np.array([[1,0],[-2,-2]])+.0
#    K=np.array([[0,-2],[1,-1]])+.0
    
    M=np.array([[m, Sa],[Sa, Ia]])
    D=np.array([[q*S*dclda/U,0],[-e*q*S*dclda/U,-q*S*c*dcmdadot]])
    K=np.array([[Kh, q*S*dclda],[0, Ka-e*q*S*dclda]])
    
    B=np.zeros([4,4])
    B[:2,:2]=M
    B[2:,2:]=np.diag(np.ones(2))

    
    A=np.zeros([4,4])
    A[:2,:2]=D
    A[:2,2:]=K
    A[2:,:2]=-np.diag(np.ones(2))
   
#    Ai=inv(A)
    ev=eig(A,-B)  
#    for i in range(len(ev[0])):
    print(np.sum(A@ev[1][:,0]+ev[0][0]*B@ev[1][:,0]))
#    N=ev[0][0]**2*M+ev[0][0]*D+K
#    print(eig(N))         
#        print(ev[i,3:]
    return ev[0]
    
# theodorsen garrik    
def flutter_theodorsen_garrik():
    return 0   



#get_Ud()
get_flutter(plot=True)

#generate_lookup_clalph(Umin,Umax,steps,l,c,x_ea,x_ac,x_sp,k,filename,rho=1.2,kin_visc=15.11E-6)
#get_Ud()






