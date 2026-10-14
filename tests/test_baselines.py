"""Independent properties of the new stateful controllers and optimization."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from quad_dac.configuration import resolve
from quad_dac.baselines import PID,DOB,L1,NMPC,AugmentedDisturbanceObserver,allocate
from quad_dac.models import Model
from quad_dac.plant import Params,transition
from quad_dac.experiment import run
from quad_dac.references import reference

C=resolve(json.loads((Path(__file__).parents[1]/'configs/default.json').read_text()))

def setup():
    c=copy.deepcopy(C); p=Params(**c['nominal_params']); m=Model('nonlinear',p,c['dt'],c['substeps'])
    return c,p,m

class BaselineTests(unittest.TestCase):
    def test_pid_integral_sign_and_antiwindup(self):
        c,p,m=setup(); ctl=PID('pid',m,c); x=np.array([0.,2.,0.,0.,0.,0.]); r=np.array([.1,2.,0.,0.,0.,0.])
        _,v,_=ctl.action(x,r,{},0); np.testing.assert_array_equal(v,[0.,0.]); ctl.update({},1)
        self.assertGreater(ctl.integral[0],0.)
        ctl.integral[:]=0; r[0]=100
        for k in range(50): ctl.action(x,r,{},k); ctl.update({},k+1)
        np.testing.assert_array_equal(ctl.integral,[0.,0.])
    def test_observer_constant_disturbance(self):
        c,p,m=setup(); observer=AugmentedDisturbanceObserver(m,c['dob'])
        x=np.array([0.,2.,0.,0.,0.,0.]); u=m.hover; d=np.array([.4,-.2,.1])
        for k in range(120):
            xn=transition(x,u,p,m.dt,m.substeps,d=d); observer.observe(x,u,xn); x=xn
        np.testing.assert_allclose(observer.disturbance,d,atol=.005)
        self.assertGreater(np.linalg.eigvalsh(observer.P).min(),-1e-12)
    def test_l1_unmatched_channel_not_canceled_as_thrust(self):
        c,p,m=setup(); ctl=L1('l1',m,c); ctl.h=np.array([.7,0.,0.])
        x=np.array([0.,2.,0.,0.,0.,0.]); r=np.array([0.,2.,0.,0.,0.,0.])
        ctl.action(x,r,{},0)
        np.testing.assert_allclose(ctl.extra_input,[0.,0.],atol=1e-12)
        self.assertAlmostEqual(ctl.sigma[2],p.mass*.7)
        self.assertAlmostEqual(np.linalg.det(ctl.channels(.4)),np.linalg.det(ctl.channels(0.)))
    def test_l1_predictor_constant_error_solution(self):
        c,p,m=setup(); ctl=L1('l1',m,c)
        x=np.array([0.,2.,0.,0.,0.,0.]); u=m.hover; d=np.array([.4,-.2,0.])
        ctl.pred=x[[2,3,5]].copy()
        xn=transition(x,u,p,m.dt,m.substeps,d=d); ctl.observe(x,u,xn)
        np.testing.assert_allclose(ctl.pred-xn[[2,3,5]],-ctl.Phi*d,atol=1e-12)
        np.testing.assert_allclose(ctl.h,ctl.E*d,atol=1e-12)
        # Causal estimates cannot be used until transition has been observed.
        self.assertGreater(ctl.h[0],0.)
    def test_nmpc_gradient_and_descent(self):
        c,p,m=setup(); c['mpc'].update(horizon=7,move_block=2,max_iterations=12)
        ctl=NMPC('nmpc',m,c); x=np.array([.1,2.1,.1,-.1,.08,.02])
        refs=np.array([reference(k*m.dt,c['reference']) for k in range(ctl.N)])
        target=ctl.target(ctl.N*m.dt); flat=np.tile(m.hover,ctl.blocks)+np.linspace(-.1,.1,ctl.blocks*2)
        d=np.array([.2,-.1,.03]); value,g=ctl.objective(flat,x,refs,target,d)
        eps=1e-6; eye=np.eye(len(flat))
        numerical=np.array([(ctl.objective(flat+eps*e,x,refs,target,d)[0]-ctl.objective(flat-eps*e,x,refs,target,d)[0])/(2*eps) for e in eye])
        np.testing.assert_allclose(g,numerical,rtol=2e-5,atol=1e-6)
        u,_,_=ctl.action(x,refs[0],{},0)
        self.assertTrue(np.all((u>=0)&(u<=p.rotor_max)))
        self.assertLessEqual(ctl.last_final_cost,ctl.last_initial_cost+1e-12)
    def test_all_baselines_hover_and_initialization(self):
        for name,cls in [('pid',PID),('dob',DOB),('l1',L1),('nmpc',NMPC)]:
            c,p,m=setup(); c['reference']['kind']='hover'; ctl=cls(name,m,c)
            x=np.array([0.,2.,0.,0.,0.,0.]); r=reference(0,c['reference'])
            u,v,_=ctl.action(x,r,{},0)
            np.testing.assert_allclose(u,m.hover,atol=1e-8,err_msg=name)
    def test_drag_sensitivity(self):
        p=Params(drag=.2,angular_drag=.003); x=np.array([.2,2.,.4,-.2,.1,.15]); u=np.array([4.5,5.])
        _,A,B=transition(x,u,p,.04,jac=True); eps=1e-6
        num=np.column_stack([(transition(x+eps*e,u,p,.04)-transition(x-eps*e,u,p,.04))/(2*eps) for e in np.eye(6)])
        np.testing.assert_allclose(A,num,atol=1e-7)
    def test_no_wind_new_baselines(self):
        for name in ['pid','dob','l1','nmpc']:
            c=copy.deepcopy(C); c['scenario'].update(regimes=['none'],switches=0,duration=2.)
            data,metrics=run(c,31,name,'nonlinear')
            self.assertFalse(metrics['failed'],name)
            self.assertLess(metrics['position_rmse_observed'],.08,name)
    def test_state_resets_between_runs(self):
        c=copy.deepcopy(C); c['scenario'].update(regimes=['bias'],switches=0,duration=.4)
        for name in ['pid','dob','l1']:
            a,_=run(c,44,name,'nonlinear'); b,_=run(c,44,name,'nonlinear')
            np.testing.assert_array_equal(a['u'],b['u'])

if __name__=='__main__': unittest.main()
