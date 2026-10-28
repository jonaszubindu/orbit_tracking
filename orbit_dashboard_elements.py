import numpy as np
from orbit_toy import MU

def elements(states):
    s=np.atleast_2d(states); r=s[:,:3]; v=s[:,3:]
    R=np.linalg.norm(r,axis=1); h=np.cross(r,v); H=np.linalg.norm(h,axis=1)
    n=np.cross(np.array([0.,0.,1.]),h); N=np.linalg.norm(n,axis=1)
    ev=np.cross(v,h)/MU-r/R[:,None]; e=np.linalg.norm(ev,axis=1)
    a=-MU/(np.sum(v*v,axis=1)-2*MU/R)
    inc=np.degrees(np.arccos(np.clip(h[:,2]/H,-1,1)))
    Om=np.degrees(np.arctan2(n[:,1],n[:,0]))%360
    w=np.degrees(np.arctan2(np.sum(np.cross(n,ev)*h,axis=1)/(N*e*H+1e-30),
                             np.sum(n*ev,axis=1)/(N*e+1e-30)))%360
    f=np.arctan2(np.sum(np.cross(ev,r)*h,axis=1)/(e*R*H+1e-30),
                 np.sum(ev*r,axis=1)/(e*R+1e-30))
    E=2*np.arctan2(np.sqrt(np.maximum(1-e,0))*np.sin(f/2),np.sqrt(1+e)*np.cos(f/2))
    M=np.degrees((E-e*np.sin(E))%(2*np.pi))
    Om[N<1e-8]=np.nan; w[(N<1e-8)|(e<1e-7)]=np.nan; M[e<1e-7]=np.nan
    return np.column_stack((a,e,inc,Om,w,M))

