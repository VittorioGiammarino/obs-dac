"""Backward-compatible defaults; resolved values are always written to disk."""
import copy

NEW_DEFAULTS={
 'pid':{'ki':[.8,.8]},
 'dob':{'process_density':[.3,.3,1.], 'measurement_variance':[.0004,.0004,.001],
        'initial_disturbance_variance':[1.,1.,4.], 'estimate_bound':[8.,8.,30.], 'torque_bound':.2},
 'l1':{'predictor_rates':[5.,5.,5.], 'filter_bandwidth':[10.,10.],
       'estimate_bound':[8.,8.,30.], 'input_bounds':[2.,.2]},
 'mpc':{'horizon':20,'move_block':2,'max_iterations':35,'ftol':1e-8,'gtol':1e-5,'terminal_scale':1.},
 'transition_window':2.
}

def resolve(cfg):
    c=copy.deepcopy(cfg)
    for k,v in NEW_DEFAULTS.items():
        if isinstance(v,dict): c[k]={**v,**c.get(k,{})}
        else: c.setdefault(k,v)
    if c['dt']<=0 or c['substeps']<1: raise ValueError('Invalid integration settings')
    if c['mpc']['horizon']<1 or c['mpc']['move_block']<1: raise ValueError('Invalid MPC horizon')
    if c['true_params']['rotor_max']!=c['nominal_params']['rotor_max']:
        raise ValueError('Known rotor limits must match; model-error sweeps must not alter physical authority')
    return c
