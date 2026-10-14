"""Stateful control baselines. No simulator parameters or disturbance access.

L1 is a planar matched-channel adaptation of Wu et al., arXiv:2302.07208v2,
Eqs. 19--25, with a PID base and a sampled predictor. No inherited guarantees.
"""
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
from .controllers import nominal, stage
from .estimation import history
from .plant import transition
from .references import reference
from .models import Model

VEL = np.array([2,3,5])


def allocate(T, tau, p):
    raw=np.array([(T-tau/p.arm)/2,(T+tau/p.arm)/2])
    return np.clip(raw,0,p.rotor_max), bool(np.any((raw<0)|(raw>p.rotor_max)))


def velocity_increment(model,x,u):
    return model.step(x,u)[VEL]-x[VEL]


class AugmentedDisturbanceObserver:
    """Kalman disturbance observer with nonlinear known velocity increments.

    State [vx,vz,omega, dx,dz,domega]. Disturbances follow a random walk.
    Uses applied commands and observed x only. Measurement covariance is a
    design regularizer in this noise-free benchmark, not injected sensor noise.
    """
    def __init__(self,model,c):
        self.model=model; self.c=c; dt=model.dt
        self.F=np.block([[np.eye(3),dt*np.eye(3)],[np.zeros((3,3)),np.eye(3)]])
        self.C=np.c_[np.eye(3),np.zeros((3,3))]
        # Discretized continuous white disturbance-rate process covariance.
        q=np.diag(c['process_density'])
        self.Q=np.block([[dt**3*q/3,dt**2*q/2],[dt**2*q/2,dt*q]])
        self.R=np.diag(c['measurement_variance'])
        self.P=np.diag([*c['measurement_variance'],*c['initial_disturbance_variance']])
        self.mean=None 
        self.bound=np.asarray(c['estimate_bound'])

    @property
    def disturbance(self):
        return np.zeros(3) if self.mean is None else self.mean[3:].copy()
    
    def observe(self,x,u,xn):

        if self.mean is None: 
            self.mean=np.r_[x[VEL],np.zeros(3)]

        predicted=self.F@self.mean
        predicted[:3]+=velocity_increment(self.model,x,u)
        P=self.F@self.P@self.F.T+self.Q
        K=np.linalg.solve(self.C@P@self.C.T+self.R,self.C@P).T
        self.mean=predicted+K@(xn[VEL]-self.C@predicted)
        self.mean[3:]=np.clip(self.mean[3:],-self.bound,self.bound)
        J=np.eye(6)-K@self.C
        self.P=J@P@J.T+K@self.R@K.T
        self.P=(self.P+self.P.T)/2


class Base:
    def __init__(self,method,model,cfg):
        self.method=method 
        self.model=model 
        self.cfg=cfg
        self.M=np.zeros((0,2,2)) 
        self.grad_norm=0.0 
        self.skipped=0
        self.d_est=np.zeros(3) 
        self.extra_input=np.zeros(2)
        self.solver_success=True 
        self.solver_iterations=0 
        self.solver_fallback=False

    def observe(self,x,u,xn): 
        pass

    def update(self,record,end): 
        pass

    def diagnostics(self):
        return dict(disturbance_estimate=self.d_est.copy(),
                    extra_input=self.extra_input.copy(),
                    solver_success=float(self.solver_success),
                    solver_iterations=self.solver_iterations,
                    solver_fallback=float(self.solver_fallback))


class PID(Base):
    """Outer-loop PI(D on measured velocity), common attitude PD.

    Integral acceleration bdot=Ki*(p_ref-p); box projection and conditional
    integration on infeasible desired tilt/thrust/rotor requests.
    """
    def __init__(self,method,model,cfg):
        super().__init__(method,model,cfg); self.integral=np.zeros(2); self.pending=np.zeros(2)
        self.freeze=False
    def action(self,x,r,record,t):
        v=self.integral.copy()
        u,info = nominal(x,r,v,self.model.p,self.cfg['controller'])
        c=self.cfg['controller'] 
        a=r[4:6]+np.array(c['kp'])*(r[:2]-x[:2])+np.array(c['kd'])*(r[2:4]-x[2:4])+v
        self.freeze=(info['saturated'] or a[1]+self.model.p.gravity<=c['min_vertical_accel'] or
                     abs(np.arctan2(-a[0],a[1]+self.model.p.gravity))>=c['tilt_max'])
        self.pending=self.model.dt*np.asarray(self.cfg['pid']['ki'])*(r[:2]-x[:2])
        return u,v,info
    
    def update(self,record,end):
        delta=self.pending.copy()
        # Under allocation saturation permit only increments which unwind stored integral.
        if self.freeze: delta=np.where(self.integral*delta<0,delta,0.)
        self.integral=np.clip(self.integral+delta,-self.cfg['correction_bound'],self.cfg['correction_bound'])


class DOB(Base):
    def __init__(self,method,model,cfg):
        super().__init__(method, model, cfg); 
        self.observer=AugmentedDisturbanceObserver(model,cfg['dob'])

    def action(self,x,r,record,t):
        self.d_est=self.observer.disturbance
        v=np.clip(-self.d_est[:2],-self.cfg['correction_bound'],self.cfg['correction_bound'])
        # Outer correction handles translation; matched angular correction handles model torque error.
        u,info=nominal(x,r,v,self.model.p,self.cfg['controller'])
        tau=-self.model.p.inertia*self.d_est[2]
        tau=np.clip(tau,-self.cfg['dob']['torque_bound'],self.cfg['dob']['torque_bound'])
        u,sat=allocate(u.sum(),self.model.p.arm*(u[1]-u[0])+tau,self.model.p)
        info['saturated']|=sat; self.extra_input=np.array([0.,tau])
        return u,v,info
    
    def observe(self,x,u,xn): 
        self.observer.observe(x,u,xn)

class DOB_DAC(Base):
    """DOB cancellation + learned residual disturbance-action correction.

    Replay holds recorded disturbance estimates and model residuals fixed.
    Gradients therefore omit the observer's counterfactual evolution.
    """

    def __init__(self, method, model, cfg):
        super().__init__(method, model, cfg)
        self.observer = AugmentedDisturbanceObserver(model, cfg['dob'])
        self.M = np.zeros((cfg['dac']['history'], 2, 2))
        self.grad_norm = 0.0
        self.skipped = 0
        self._dob_history = {}

    def _policy(self, x, r, M, z, d, jac=False):
        """Shared action/replay policy, including both saturation stages."""
        p = self.model.p
        bound = self.cfg['correction_bound']

        raw = -d[:2] + np.einsum('hij,hj->i', M, z)
        v = np.clip(raw, -bound, bound)

        if jac:
            u, C, D, info = nominal(x, r, v, p, self.cfg['controller'], jac=True)

            # Partial derivative of v with respect to flattened M.
            V = np.zeros((2,) + M.shape)
            for i in range(2):
                V[i, :, i, :] = z

            V = V.reshape(2, -1)
            V *= (np.abs(raw) < bound)[:, None]

        else:
            u, info = nominal(x, r, v, p, self.cfg['controller'])

        tau = np.clip(
            -p.inertia * d[2],
            -self.cfg['dob']['torque_bound'],
            self.cfg['dob']['torque_bound'],
        )

        # Reproduce DOB's allocation after nominal rotor clipping.
        T = u.sum()
        total_tau = p.arm * (u[1] - u[0]) + tau
        raw_u = np.array([
            (T - total_tau / p.arm) / 2,
            (T + total_tau / p.arm) / 2,
        ])
        u, saturated = allocate(T, total_tau, p)
        info['saturated'] |= saturated

        if jac:
            # With frozen d, angular correction has zero derivative.
            # Reallocation is identity in the nominal rotor commands,
            # followed by this second saturation.
            active = ((raw_u > 0) & (raw_u < p.rotor_max))
            C = active[:, None] * C
            UM = active[:, None] * (D @ V)
            return u, C, UM

        return u, v, info, tau

    def action(self, x, r, record, t):
        self.d_est = self.observer.disturbance.copy()
        self._dob_history[t] = self.d_est.copy()

        z = history(record['z'], t, len(self.M))
        u, v, info, tau = self._policy(x, r, self.M, z, self.d_est)
        self.extra_input = np.array([0.0, tau])
        return u, v, info

    def observe(self, x, u, xn):
        self.observer.observe(x, u, xn)

    def surrogate(self, M, record, end):
        """Frozen-observer replay loss and analytic gradient."""
        dc = self.cfg['dac']
        start = max(0, end - dc['replay'])
        x = record['x'][start].copy()
        S = np.zeros((6, M.size))
        loss = 0.0
        grad = np.zeros(M.size)
        hover = np.full(2, self.cfg['cost']['hover_rotor'])
        dt = self.model.dt

        for k in range(start, end):
            z = history(record['z'], k, len(M))
            d = self._dob_history[k]

            u, C, UM = self._policy(
                x, record['ref'][k], M, z, d, jac=True
            )
            US = C @ S + UM

            value, gx, gu = stage(
                x, u, record['ref'][k], self.cfg['cost'], hover
            )
            loss += dt * value
            grad += dt * (gx @ S + gu @ US)

            pred, A, B = self.model.step(x, u, True)
            x = pred + record['w'][k]
            S = A @ S + B @ US

        reg = dc['regularization']
        loss += 0.5 * reg * np.sum(M * M)
        grad += reg * M.ravel()
        return loss, grad.reshape(M.shape)

    def update(self, record, end):
        dc = self.cfg['dac']
        if end % dc['update_every']:
            return

        _, g = self.surrogate(self.M, record, end)
        if not np.all(np.isfinite(g)):
            self.skipped += 1
            return

        self.grad_norm = float(np.linalg.norm(g))
        g *= min(1.0, dc['gradient_clip'] /
                 max(self.grad_norm, 1e-15))

        self.M -= dc['learning_rate'] * g
        self.M *= min(1.0, dc['parameter_radius'] /
                      max(float(np.linalg.norm(self.M)), 1e-15))


class L1(PID):
    """PID + physical-channel L1 predictor/adaptation/filter.

    Estimate all 3 velocity-rate discrepancies; directly compensate only the
    thrust-axis force and pitch moment. No fictitious instantaneous lateral input.
    """
    def __init__(self,method,model,cfg):
        super().__init__(method,model,cfg)
        self.pred=None; self.h=np.zeros(3); self.filtered=np.zeros(2)
        self.a=np.asarray(cfg['l1']['predictor_rates']); self.E=np.exp(-self.a*model.dt)
        self.Phi=-np.expm1(-self.a*model.dt)/self.a
        self.sigma=np.zeros(3)
    def channels(self,theta):
        p=self.model.p
        # Columns: matched thrust (N), matched torque (Nm), unmatched lateral force (N).
        return np.array([[-np.sin(theta)/p.mass,0.,np.cos(theta)/p.mass],
                         [np.cos(theta)/p.mass,0.,np.sin(theta)/p.mass],
                         [0.,1/p.inertia,0.]])
    def action(self,x,r,record,t):
        if self.pred is None: self.pred=x[VEL].copy()
        self.sigma=np.linalg.solve(self.channels(x[4]),self.h)
        alpha=np.exp(-np.asarray(self.cfg['l1']['filter_bandwidth'])*self.model.dt)
        self.filtered=alpha*self.filtered+(1-alpha)*self.sigma[:2]
        p=self.model.p; bounds=np.asarray(self.cfg['l1']['input_bounds'])
        self.filtered=np.clip(self.filtered,-bounds,bounds)
        u,v,info=super().action(x,r,record,t)
        self.extra_input=-self.filtered.copy()
        u,sat=allocate(u.sum()+self.extra_input[0],p.arm*(u[1]-u[0])+self.extra_input[1],p)
        info['saturated']|=sat; self.freeze|=sat
        self.d_est=self.h.copy()
        return u,v,info
    def observe(self,x,u,xn):
        # Exact diagonal error-feedback integration assuming observed velocity is
        # linearly interpolated over the step and nominal acceleration is its
        # nominal interval average. Interpolation is causal: performed after xn.
        inc=velocity_increment(self.model,x,u)
        error=self.pred-x[VEL]
        error_next=self.E*error+self.Phi*(inc/self.model.dt+self.h-(xn[VEL]-x[VEL])/self.model.dt)
        self.pred=xn[VEL]+error_next
        self.h=-self.E/self.Phi*error_next
        self.h=np.clip(self.h,-np.asarray(self.cfg['l1']['estimate_bound']),np.asarray(self.cfg['l1']['estimate_bound']))


class NMPC(Base):
    """Direct rotor-input NMPC with augmented-state disturbance observer.

    Box-constrained single shooting, analytic RK4 sensitivities and discrete
    adjoint gradient. Terminal hover-LQR quadratic is a design heuristic.
    No hard state constraints / recursive feasibility or stability claim.
    """
    def __init__(self, method, model, cfg):
        super().__init__(method,model,cfg)
        if model.kind!='nonlinear': 
            raise ValueError('NMPC requires nonlinear model')
        
        self.observer=AugmentedDisturbanceObserver(model,cfg['dob']) 
        self.plan=None
        self.c=cfg['mpc'] 
        self.N=self.c['horizon'] 
        self.block=self.c['move_block']
        self.blocks=(self.N+self.block-1)//self.block
        linear=Model('linear',model.p,model.dt,model.substeps)
        q=np.diag([cfg['cost']['position']]*2+[cfg['cost']['velocity']]*2+[0.,0.])*model.dt
        R=np.eye(2)*cfg['cost']['effort']*model.dt
        self.P=solve_discrete_are(linear.A,linear.B,q,R)*self.c['terminal_scale']
        self.last_initial_cost=np.nan; self.last_final_cost=np.nan

    def target(self,time):
        r=reference(time,self.cfg['reference']) 
        h=self.model.dt
        jerk=(reference(time+h,self.cfg['reference'])[4:6]-reference(time-h,self.cfg['reference'])[4:6])/(2*h)
        q=r[4:6]+np.array([0.,self.model.p.gravity])
        theta=np.arctan2(-q[0],q[1]) 
        omega=(q[0]*jerk[1]-q[1]*jerk[0])/max(q@q,1e-12)
        return np.r_[r[:4],theta,omega]
    
    def objective(self,flat,x,refs,target,dist):
        U=flat.reshape(self.blocks,2) 
        state=x.copy() 
        terms=[] 
        loss=0.
        hover=np.full(2,self.cfg['cost']['hover_rotor']); dt=self.model.dt
        for k in range(self.N):
            u=U[k//self.block]
            val,gx,gu=stage(state,u,refs[k],self.cfg['cost'],hover)
            pred,A,B=transition(state,u,self.model.p,dt,self.model.substeps,d=dist,jac=True)
            terms.append((A,B,dt*gx,dt*gu)); loss+=dt*val; state=pred
        e=state-target
        loss+=e@self.P@e; adj=2*self.P@e; grad=np.zeros_like(U)
        for k in reversed(range(self.N)):
            A,B,gx,gu=terms[k]; grad[k//self.block]+=gu+B.T@adj; adj=gx+A.T@adj
        return float(loss),grad.ravel()
    
    def action(self,x,r,record,t):
        self.d_est=self.observer.disturbance
        time=t*self.model.dt 
        refs=np.array([reference(time+k*self.model.dt,self.cfg['reference']) for k in range(self.N)])
        target=self.target(time+self.N*self.model.dt)
        # Both a shifted warm start and a fresh nominal rollout are considered.
        state=x.copy() 
        nominal_inputs=[]

        for k in range(self.N):
            v=np.clip(-self.d_est[:2],-self.cfg['correction_bound'],self.cfg['correction_bound'])
            u,_=nominal(state,refs[k],v,self.model.p,self.cfg['controller'])
            nominal_inputs.append(u)
            state=transition(state,u,self.model.p,self.model.dt,self.model.substeps,d=self.d_est)

        cold=np.array(nominal_inputs)[::self.block][:self.blocks]
        fun=lambda z:self.objective(z,x,refs,target,self.d_est)
        candidates=[cold.ravel()]
        if self.plan is not None:
            full=np.repeat(self.plan,self.block,axis=0)[:self.N]
            shifted=np.vstack([full[1:],full[-1]])
            candidates.append(shifted[::self.block][:self.blocks].ravel())

        vals=[fun(z)[0] for z in candidates]; initial=candidates[int(np.argmin(vals))]
        initial_val=min(vals); self.last_initial_cost=initial_val
        best=[initial_val,initial.copy()]
        def objective(z):
            value,gradient=fun(z)
            if np.isfinite(value) and np.all(np.isfinite(gradient)) and value<best[0]: best[:]=[value,z.copy()]
            return value,gradient
        try:
            result=minimize(objective,initial,jac=True,method='L-BFGS-B',
                            bounds=[(0,self.model.p.rotor_max)]*len(initial),
                            options={
                                'maxiter': self.c['max_iterations'],
                                'maxfun': self.c['max_function_evaluations'],
                                'maxls': self.c['max_line_search'],
                                'ftol': self.c['ftol'],
                                'gtol': self.c['gtol'],
                                },
                            )
            self.solver_success=bool(result.success); self.solver_iterations=int(result.nit)
            self.solver_fallback=not self.solver_success
        except (ValueError,FloatingPointError,np.linalg.LinAlgError):
            self.solver_success=False; self.solver_iterations=0; self.solver_fallback=True
        self.last_final_cost=best[0]; self.plan=best[1].reshape(self.blocks,2)
        u=self.plan[0].copy()
        # Direct NMPC has no outer acceleration correction or controller-defined attitude reference.
        info={'theta_des':self.target(time)[4], 'saturated':bool(np.any((u<1e-6)|(u>self.model.p.rotor_max-1e-6)))}
        return u,np.zeros(2),info
    def observe(self,x,u,xn): self.observer.observe(x,u,xn)
    def diagnostics(self):
        return dict(super().diagnostics(),mpc_initial_objective=self.last_initial_cost,mpc_final_objective=self.last_final_cost)


def make_controller(method,model,cfg):
    classes={'pid':PID,'dob':DOB,'l1':L1,'nmpc':NMPC, 'dob_dac': DOB_DAC}
    if method in classes: 
        return classes[method](method,model,cfg)
    
    from .controllers import ResidualController
    if method not in ['nominal','bias','dac']: 
        raise ValueError('Unknown method '+method)
    
    return ResidualController(method,model,cfg)
