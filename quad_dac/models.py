import numpy as np
from scipy.linalg import expm
from .plant import transition, rhs_jac

class Model:
    def __init__(self,kind,p,dt,substeps):
        self.kind,self.p,self.dt,self.substeps=kind,p,dt,substeps
        self.hover=np.full(2,p.mass*p.gravity/2)
        A,B=rhs_jac(np.zeros(6),self.hover,p)
        C=np.zeros((8,8)); C[:6,:6]=A; C[:6,6:]=B
        E=expm(C*dt); self.A,self.B=E[:6,:6],E[:6,6:]
        
    def step(self,x,u,jac=False):
        if self.kind=='nonlinear':
            return transition(x,u,self.p,self.dt,self.substeps,jac=jac)
        y=self.A@x+self.B@(u-self.hover)
        return (y,self.A,self.B) if jac else y
