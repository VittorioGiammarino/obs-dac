import argparse, copy, hashlib, importlib.metadata, json, platform, subprocess, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .plant import Params,transition
from .models import Model
from .estimation import estimate
from .controllers import stage
from .baselines import make_controller
from .configuration import resolve
from .scenarios import generate
from .metrics import summarize

METHODS=[('nominal','nonlinear'),('bias','linear'),('bias','nonlinear'),('dac','linear'),('dac','nonlinear')]

def run(cfg,seed,method,kind,scenario=None):
    cfg=resolve(cfg)
    d,refs,labels,x,order=generate(cfg,seed) if scenario is None else scenario
    x=x.copy(); initial_state=x.copy(); true=Params(**cfg['true_params']); nom=Params(**cfg['nominal_params'])
    model=Model(kind,nom,cfg['dt'],cfg['substeps']); ctrl=make_controller(method,model,cfg)
    N=len(d); rec={'x':[x.copy()],'w':[],'z':[],'ref':refs}
    data={k:np.full((N,*shape),np.nan) for k,shape in
          {'x':(6,),'u':(2,),'v':(2,),'z':(2,),'w':(6,),'timing':(4,),
           'stage':(),'effort':(),'saturated':(),'attitude_error':(),
           'grad_norm':(),'parameter_norm':(),'violation':(),
           'disturbance_estimate':(3,),'extra_input':(2,), 'solver_success':(),
           'solver_iterations':(),'solver_fallback':(), 'mpc_initial_objective':(),'mpc_final_objective':()}.items()}
    data.update(ref=refs[:-1],disturbance=d,labels=labels)
    failed=False; fail_step=None
    for t in range(N):
        t0=time.perf_counter(); u,v,info=ctrl.action(x,refs[t],rec,t); diagnostics=ctrl.diagnostics(); t1=time.perf_counter()
        xn=transition(x,u,true,cfg['dt'],cfg['substeps'],d[t])
        t2=time.perf_counter(); w,z=estimate(model,x,u,xn)
        finite=bool(np.isfinite(xn).all())
        if finite: ctrl.observe(x,u,xn)
        t3=time.perf_counter()
        rec['w'].append(w); rec['z'].append(z); rec['x'].append(xn.copy())
        if finite: ctrl.update(rec,t+1)
        t4=time.perf_counter()
        violation=bool(abs(xn[4])>cfg['failure']['pitch'] or xn[1]<cfg['failure']['min_altitude'] or
                       np.linalg.norm(xn[:2]-refs[t+1,:2])>cfg['failure']['position_error'] or not np.isfinite(xn).all())
        val=stage(x,u,refs[t],cfg['cost'],np.full(2,cfg['cost']['hover_rotor']))[0]
        vals=dict(x=x,u=u,v=v,z=z,w=w,timing=[t1-t0,t3-t2,t4-t3,(t1-t0)+(t4-t2)],
                  stage=val,effort=np.sum((u-cfg['cost']['hover_rotor'])**2),saturated=info['saturated'],
                  attitude_error=np.arctan2(np.sin(info['theta_des']-x[4]),np.cos(info['theta_des']-x[4])),
                  grad_norm=ctrl.grad_norm,parameter_norm=np.linalg.norm(ctrl.M),violation=violation)
        vals.update(diagnostics)
        for k,value in vals.items(): data[k][t]=value
        x=xn
        if violation: failed=True; fail_step=t+1; break
    metrics=summarize(data,cfg,failed,fail_step)
    if method=='nmpc':
        valid=np.isfinite(data['solver_success'])
        metrics['solver_nonconvergence_frequency']=float(np.mean(data['solver_success'][valid]<.5))
        metrics['solver_fallback_count']=int(np.sum(data['solver_fallback'][valid]))
        metrics['solver_mean_iterations']=float(np.mean(data['solver_iterations'][valid]))
    metrics.update(seed=seed,method=method,model=kind,skipped_updates=ctrl.skipped,
                   scenario_hash=hashlib.sha256(d.tobytes()+refs.tobytes()+initial_state.tobytes()+json.dumps(cfg['true_params'],sort_keys=True).encode()).hexdigest(),order=order,
                   final_parameters=ctrl.M.tolist())
    return data,metrics

def variants(cfg):
    yield 'base',cfg
    for case in cfg.get('cases',[]):
        c=copy.deepcopy(cfg)
        for path,value in case['set'].items():
            target=c; parts=path.split('.')
            for part in parts[:-1]: target=target[part]
            target[parts[-1]]=value
        yield case['name'],resolve(c)
    for axis,values in cfg.get('sweeps',{}).items():
        for value in values:
            c=copy.deepcopy(cfg)
            if axis=='demand': c['reference']['amplitude']*=value
            elif axis=='mismatch':
                # Keep common nominal gains, true plant and primary cost fixed.
                c['nominal_params']['mass']*=value
            elif axis=='amplitude': c['scenario']['amplitude']=(np.array(c['scenario']['amplitude'])*value).tolist()
            elif axis=='structure': c['scenario']['regimes']=[value]
            else: raise ValueError(axis)
            yield f'{axis}_{value}',c

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.json'); p.add_argument('--output',required=True)
    p.add_argument('--methods',nargs='+',help='e.g. pid:nonlinear dac:linear nmpc:nonlinear')
    p.add_argument('--seeds',nargs='+',type=int); p.add_argument('--variant',default=None)
    a=p.parse_args(); cfg=resolve(json.loads(Path(a.config).read_text())); root=Path(a.output)
    if a.methods: cfg['methods']=[item.split(':') for item in a.methods]
    if a.seeds is not None: cfg['seeds']=a.seeds
    if a.variant: cfg['selected_variant']=a.variant
    root.mkdir(parents=True,exist_ok=False)
    versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','matplotlib']}
    try: rev=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError): rev=None
    metadata={'python':platform.python_version(),'platform':platform.platform(),'dependencies':versions,'git_revision':rev,
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (root/'config.json').write_text(json.dumps(cfg,indent=2))
    for name,c in variants(cfg):
        if a.variant is not None and name!=a.variant: continue
        dest=root/name; dest.mkdir(); (dest/'config.json').write_text(json.dumps(c,indent=2))
        for seed in c['seeds']:
            scenario=generate(c,seed)
            for method,kind in c.get('methods',METHODS):
                data,m=run(c,seed,method,kind,scenario); m['variant']=name
                prefix=dest/f'{seed}_{method}_{kind}'
                prefix.with_suffix('.json').write_text(json.dumps(m,indent=2,allow_nan=False))
                if c.get('save_timeseries',True): np.savez_compressed(prefix.with_suffix('.npz'),**data)
                print(name,seed,method,kind,'failed=',m['failed'],flush=True)

if __name__=='__main__': main()
