import copy,json,unittest
from pathlib import Path
import numpy as np
from quad_dac.plant import Params,transition
from quad_dac.controllers import nominal,surrogate,ResidualController
from quad_dac.models import Model
from quad_dac.estimation import estimate,history
from quad_dac.experiment import run
from quad_dac.scenarios import generate
from quad_dac.metrics import recoveries

CFG=json.loads((Path(__file__).parents[1]/'configs/default.json').read_text())
def fd(f,x,h=1e-6):
    return np.column_stack([(f(x+np.eye(len(x))[j]*h)-f(x-np.eye(len(x))[j]*h))/(2*h) for j in range(len(x))])

class BenchmarkTests(unittest.TestCase):
    def test_hover(self):
        p=Params(); x=np.array([0.,2.,0.,0.,0.,0.]); u=np.ones(2)*p.mass*p.gravity/2
        np.testing.assert_allclose(transition(x,u,p,.04),x,atol=1e-14)
        for kind in ['linear','nonlinear']:
            np.testing.assert_allclose(Model(kind,p,.04,2).step(x,u),x,atol=1e-14)
    def test_rotor_limits_and_controller_jacobian(self):
        p=Params(); c=CFG['controller']; r=np.array([.1,2.,.1,0.,.1,0.]); x=np.array([0.,2.,0.,0.,.02,.1]); v=np.array([.1,.1])
        u,A,B,_=nominal(x,r,v,p,c,True)
        np.testing.assert_allclose(np.c_[A,B],fd(lambda q:nominal(q[:6],r,q[6:],p,c)[0],np.r_[x,v]),atol=1e-7)
        for q in [np.ones(6)*100,-np.ones(6)*100]:
            u,_=nominal(q,r,v,p,c); T=u.sum(); tau=p.arm*(u[1]-u[0])
            self.assertTrue(np.all((u>=0)&(u<=p.rotor_max)))
            self.assertLessEqual(abs(tau),p.arm*min(T,2*p.rotor_max-T)+1e-12)
    def test_integrator_and_sensitivity(self):
        x=np.array([0.,2.,.2,0.,.2,.3]); u=np.array([4.,6.]); p=Params()
        y,A,B=transition(x,u,p,.08,2,jac=True)
        np.testing.assert_allclose(np.c_[A,B],fd(lambda q:transition(q[:6],q[6:],p,.08,2),np.r_[x,u]),atol=1e-7)
        fine=transition(x,u,p,.08,64)
        self.assertLess(np.linalg.norm(y-fine),np.linalg.norm(transition(x,u,p,.08,1)-fine))
        self.assertLess(np.linalg.norm(y-fine),1e-5)
    def test_linearization(self):
        p=Params(); model=Model('linear',p,.04,2); x=np.array([0.,2.,0.,0.,0.,0.]); u=model.hover
        _,A,B=transition(x,u,p,.04,2,jac=True)
        np.testing.assert_allclose(A,model.A,atol=1e-12); np.testing.assert_allclose(B,model.B,atol=1e-12)
    def test_estimator_and_causality(self):
        p=Params(); model=Model('nonlinear',p,.04,2); x=np.array([0.,2.,0.,0.,.2,.3]); u=np.array([4.,6.]); d=np.array([.4,-.2])
        xn=transition(x,u,p,.04,2,d); w,z=estimate(model,x,u,xn)
        np.testing.assert_allclose(z,d,atol=1e-12)
        np.testing.assert_allclose(w[:2],.5*.04**2*d,atol=1e-12)
        np.testing.assert_allclose(history([np.array([1.,2.]),np.array([99.,99.])],1,3),[[1,2],[0,0],[0,0]])
        ctl=ResidualController('dac',model,CFG); rec={'z':[]}
        _,v,_=ctl.action(x,np.array([0.,2.,0.,0.,0.,0.]),rec,0)
        np.testing.assert_array_equal(v,[0,0])
    def test_replay_and_gradient(self):
        for kind in ['linear','nonlinear']:
            cfg=copy.deepcopy(CFG); cfg['dac']['history']=2; cfg['dac']['replay']=8
            model=Model(kind,Params(),cfg['dt'],2); M=np.ones((2,2,2))*.03
            x=np.array([0.,2.,0.,0.,0.,0.]); rec={'x':[x.copy()],'z':[],'w':[],'ref':[]}
            for k in range(9):
                r=np.array([.1,2.,0.,0.,0.,0.]); rec['ref'].append(r)
                v=np.einsum('hij,hj->i',M,history(rec['z'],k,2)); u,_=nominal(x,r,v,model.p,cfg['controller'])
                xn=transition(x,u,model.p,cfg['dt'],2,np.array([.5,.2])); w,z=estimate(model,x,u,xn)
                rec['w'].append(w); rec['z'].append(z); rec['x'].append(xn); x=xn
            val,g=surrogate(M,rec,model,cfg,9)
            numerical=fd(lambda v:np.array([surrogate(v.reshape(M.shape),rec,model,cfg,9)[0]]),M.ravel()).reshape(M.shape)
            np.testing.assert_allclose(g,numerical,atol=1e-7,rtol=1e-5)
            # Recorded residuals reconstruct every factual transition exactly.
            for k in range(9):
                v=np.einsum('hij,hj->i',M,history(rec['z'],k,2)); u,_=nominal(rec['x'][k],rec['ref'][k],v,model.p,cfg['controller'])
                np.testing.assert_allclose(model.step(rec['x'][k],u)+rec['w'][k],rec['x'][k+1],atol=1e-12)
    def test_no_wind_tracking(self):
        cfg=copy.deepcopy(CFG); cfg['scenario'].update(regimes=['none'],switches=0,duration=12.)
        for kind in ['hover','gentle']:
            cfg['reference']['kind']=kind; data,m=run(cfg,22,'nominal','nonlinear')
            self.assertFalse(m['failed']); self.assertLess(m['position_rmse_observed'],.08)
    def test_pairing(self):
        a=generate(CFG,100); b=generate(CFG,100)
        np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1])
        c=copy.deepcopy(CFG); c['nominal_params']['mass']=.8
        np.testing.assert_array_equal(a[0],generate(c,100)[0])
    def test_recovery_censoring(self):
        out=recoveries(np.r_[np.zeros(20),np.ones(20)],[(0,20),(20,40)],.1,dict(window=.3,persistence=.2,threshold=.2))
        self.assertAlmostEqual(out[0]['time'],.4); self.assertFalse(out[0]['censored']); self.assertTrue(out[1]['censored'])
    def test_failure_tail(self):
        c=copy.deepcopy(CFG); c['scenario'].update(regimes=['none'],switches=0,duration=1.)
        c['initial_offset'][0]=10
        data,m=run(c,0,'nominal','nonlinear')
        self.assertTrue(m['failed']); self.assertEqual(m['completed_steps'],1)
        self.assertAlmostEqual(m['cost_with_failure_penalty'],m['observed_cost']+24*c['dt']*c['failure']['tail_stage_penalty'])

if __name__=='__main__': unittest.main()
