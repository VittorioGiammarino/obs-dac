"""SI units. Rotor commands [left,right]; positive pitch tilts thrust left."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Params:
    mass: float = 1.0
    inertia: float = 0.02
    arm: float = 0.2
    gravity: float = 9.81
    rotor_max: float = 10.0
    drag: float = 0.0
    angular_drag: float = 0.0

    def __post_init__(self):
        if min(self.mass, self.inertia, self.arm, self.gravity, self.rotor_max) <= 0:
            raise ValueError("Physical parameters must be positive")
        
        if min(self.drag,self.angular_drag)<0: 
            raise ValueError("Drag must be nonnegative")

def rhs(x, u, p, d):
    T = u.sum()
    angular_d=d[2] if len(d)>2 else 0.

    return np.array([x[2], x[3], -T*np.sin(x[4])/p.mass+d[0]-p.drag*x[2]/p.mass,
                     T*np.cos(x[4])/p.mass-p.gravity+d[1]-p.drag*x[3]/p.mass, x[5],
                     p.arm*(u[1]-u[0])/p.inertia-p.angular_drag*x[5]/p.inertia+angular_d])

def rhs_jac(x, u, p):
    A, B = np.zeros((6,6)), np.zeros((6,2))
    A[0,2]=A[1,3]=A[4,5]=1
    A[2,2]=A[3,3]=-p.drag/p.mass
    A[5,5]=-p.angular_drag/p.inertia
    A[2,4]=-u.sum()*np.cos(x[4])/p.mass
    A[3,4]=-u.sum()*np.sin(x[4])/p.mass
    B[2,:]=-np.sin(x[4])/p.mass
    B[3,:]=np.cos(x[4])/p.mass
    B[5,:]=np.array([-p.arm,p.arm])/p.inertia
    return A, B

def transition(x, u, p, dt, substeps=2, d=None, jac=False):
    """RK4 with held applied rotors and acceleration disturbance; exact RK4 sensitivities."""
    d = np.zeros(2) if d is None else np.asarray(d)
    y=np.asarray(x,dtype=float).copy(); h=dt/substeps
    S=np.column_stack((np.eye(6),np.zeros((6,2))))
    U=np.column_stack((np.zeros((2,6)),np.eye(2)))

    def f(y,S):
        k=rhs(y,u,p,d)
        if not jac: return k, np.zeros_like(S)
        A,B=rhs_jac(y,u,p)
        return k,A@S+B@U
    
    for _ in range(substeps):
        k1,s1=f(y,S)
        k2,s2=f(y+h*k1/2,S+h*s1/2)
        k3,s3=f(y+h*k2/2,S+h*s2/2) 
        k4,s4=f(y+h*k3,S+h*s3)
        y=y+h*(k1+2*k2+2*k3+k4)/6
        S=S+h*(s1+2*s2+2*s3+s4)/6

    return (y,S[:,:6],S[:,6:]) if jac else y
