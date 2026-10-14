import numpy as np
from .estimation import history

def nominal(x,r,v,p,c,jac=False):
    """Memoryless cascaded PD; analytic derivatives wrt [state, correction]."""
    kp=np.asarray(c['kp']); kd=np.asarray(c['kd'])
    a=r[4:6]+kp*(r[:2]-x[:2])+kd*(r[2:4]-x[2:4])+v
    q=a+np.array([0.,p.gravity])

    # Keep desired acceleration in an upward cone's half-plane.
    Q=np.column_stack((-np.diag(kp),-np.diag(kd),np.zeros((2,2)),np.eye(2)))

    if q[1]<=c['min_vertical_accel']: 
        q[1]=c['min_vertical_accel']
        Q[1]=0

    norm=np.linalg.norm(q)
    raw_theta=np.arctan2(-q[0],q[1])
    theta=np.clip(raw_theta,-c['tilt_max'],c['tilt_max'])
    Dtheta=np.array([-q[1],q[0]])@Q/(norm*norm)

    if abs(raw_theta)>=c['tilt_max']: 
        Dtheta*=0
        
    T=p.mass*norm; DT=p.mass*q@Q/norm
    e=np.arctan2(np.sin(theta-x[4]),np.cos(theta-x[4]))
    De=Dtheta.copy(); De[4]-=1
    tau=p.inertia*(c['attitude_kp']*e-c['attitude_kd']*x[5])
    Dtau=p.inertia*c['attitude_kp']*De; Dtau[5]-=p.inertia*c['attitude_kd']
    raw=np.array([(T-tau/p.arm)/2,(T+tau/p.arm)/2])
    D=np.array([(DT-Dtau/p.arm)/2,(DT+Dtau/p.arm)/2])
    D*=((raw>0)&(raw<p.rotor_max))[:,None]
    u=np.clip(raw,0,p.rotor_max)
    info={'theta_des':theta,'saturated':bool(np.any(raw!=u))}
    return (u,D[:,:6],D[:,6:],info) if jac else (u,info)

def correction(M, z, bound):
    raw=np.einsum('hij,hj->i',M,z)
    return np.clip(raw,-bound,bound)

def stage(x,u,r,c,hover):
    ep=x[:2]-r[:2]; ev=x[2:4]-r[2:4]; du=u-hover
    val=c['position']*(ep@ep)+c['velocity']*(ev@ev)+c['effort']*(du@du)
    gx=np.r_[2*c['position']*ep,2*c['velocity']*ev,0.,0.]
    gu=2*c['effort']*du
    return val,gx,gu

def surrogate(M,record,model,cfg,end):
    """Sum dt stage costs over latest L completed transitions + L2 regularizer.
    Recorded states anchor replay, recorded residuals/features stay fixed.
    Exact forward sensitivities of the discrete replay map, including clipping.
    """
    H=cfg['dac']['history']; start=max(0,end-cfg['dac']['replay'])
    x=record['x'][start].copy(); S=np.zeros((6,M.size)); loss=0.; grad=np.zeros(M.size)
    for k in range(start,end):
        z=history(record['z'],k,H)
        raw=np.einsum('hij,hj->i',M,z); v=np.clip(raw,-cfg['correction_bound'],cfg['correction_bound'])
        V=np.zeros((2,H,2,2))
        for i in range(2): V[i,:,i,:]=z
        V=V.reshape(2,-1)* (np.abs(raw)<cfg['correction_bound'])[:,None]
        u,C,D,_=nominal(x,record['ref'][k],v,model.p,cfg['controller'],True)
        US=C@S+D@V
        val,gx,gu=stage(x,u,record['ref'][k],cfg['cost'],np.full(2,cfg['cost']['hover_rotor']))
        loss+=model.dt*val; grad+=model.dt*(gx@S+gu@US)
        pred,A,B=model.step(x,u,True)
        x=pred+record['w'][k]; S=A@S+B@US
    reg=cfg['dac']['regularization']
    return loss+.5*reg*np.sum(M*M), (grad+reg*M.ravel()).reshape(M.shape)

class ResidualController:
    def __init__(self,method,model,cfg):
        self.method,self.model,self.cfg = method,model,cfg
        self.M=np.zeros((cfg['dac']['history'], 2, 2)) 
        self.bias=np.zeros(2)
        self.grad_norm=0.0
        self.skipped=0.0

    def action(self,x,r,record,t):
        if self.method=='nominal': 
            v = np.zeros(2)

        elif self.method=='bias': 
            v = np.clip(-self.bias,-self.cfg['correction_bound'],self.cfg['correction_bound'])

        else: 
            v = correction(self.M, history(record['z'],t,len(self.M)), self.cfg['correction_bound'])

        u, info = nominal(x, r, v, self.model.p, self.cfg['controller'])

        return u,v,info
    
    def observe(self,x,u,xn): pass

    def diagnostics(self): return {}

    def update(self,record,end):
        if self.method=='bias':
            beta=1-np.exp(-self.model.dt/self.cfg['bias_time_constant'])
            self.bias=(1-beta)*self.bias+beta*record['z'][-1]
        if self.method!='dac' or end%self.cfg['dac']['update_every']: return
        _,g=surrogate(self.M,record,self.model,self.cfg,end)
        if not np.all(np.isfinite(g)): self.skipped+=1; return
        self.grad_norm=float(np.linalg.norm(g)); dc=self.cfg['dac']
        g*=min(1,dc['gradient_clip']/max(self.grad_norm,1e-15))
        self.M-=dc['learning_rate']*g
        self.M*=min(1,dc['parameter_radius']/max(np.linalg.norm(self.M),1e-15))
