import numpy as np

def recoveries(error,segments,dt,c):
    """Censored at segment end; detection time includes window and persistence."""
    W=max(1,round(c['window']/dt)); P=max(1,round(c['persistence']/dt)); out=[]
    for a,b in segments:
        detected=None; count=0
        for k in range(a+W-1,b):
            rms=np.sqrt(np.mean(error[k-W+1:k+1]**2))
            count=count+1 if rms<c['threshold'] else 0
            if count>=P: detected=(k-a+1)*dt; break
        out.append({'time':detected if detected is not None else (b-a)*dt,
                    'censored':detected is None,'minimum_measurable':(W+P-1)*dt})
    return out

def summarize(data,cfg,failed,fail_step):
    dt=cfg['dt']; err=np.linalg.norm(data['x'][:,:2]-data['ref'][:,:2],axis=1)
    verr=np.linalg.norm(data['x'][:,2:4]-data['ref'][:,2:4],axis=1)
    N=len(err); steps=max(1,round(cfg['scenario']['duration']/dt))
    segments=[(a,min(a+steps,N)) for a in range(0,N,steps)]
    completed=np.isfinite(data['stage'])
    # Missing samples carry a preregistered fixed stage penalty. Failure remains separate.
    costs=np.where(completed,data['stage'],cfg['failure']['tail_stage_penalty'])
    times=data['timing'][completed]
    out={'failed':bool(failed),'failure_step':fail_step,'completed_steps':int(completed.sum()),
         'position_rmse_observed':float(np.sqrt(np.nanmean(err**2))),
         'velocity_rmse_observed':float(np.sqrt(np.nanmean(verr**2))),
         'cost_with_failure_penalty':float(dt*costs.sum()),
         'observed_cost':float(dt*np.nansum(data['stage'])),
         'effort_observed':float(dt*np.nansum(data['effort'])),
         'attitude_tracking_rmse_observed':float(np.sqrt(np.nanmean(data['attitude_error']**2))),
         'max_abs_pitch':float(np.nanmax(np.abs(data['x'][:,4]))),
         'velocity_cost_observed':float(dt*cfg['cost']['velocity']*np.nansum(verr**2)),
         'position_cost_observed':float(dt*cfg['cost']['position']*np.nansum(err**2)),
         'effort_cost_observed':float(dt*cfg['cost']['effort']*np.nansum(data['effort'])),
         'saturation_frequency_observed':float(np.mean(data['saturated'][completed])),
         'constraint_violation_steps':int(np.nansum(data['violation'])),
         'recovery':recoveries(err,segments,dt,cfg['recovery']),
         'regimes':[]}
    out['phases']=[]
    for q,(a,b) in enumerate(segments):
        cut=min(b,a+max(1,round(cfg.get('transition_window',2.)/dt)))
        for phase,j,k in [('transition',a,cut),('later',cut,b)]:
            if j==k: continue
            ok=bool(completed[j:k].all())
            out['phases'].append(dict(segment=q,regime=str(data['labels'][a]),phase=phase,
                duration=(k-j)*dt,complete=ok,cost_with_failure_penalty=float(dt*costs[j:k].sum()),
                mean_stage_cost=float(np.mean(costs[j:k])),
                position_rmse=float(np.sqrt(np.mean(err[j:k]**2))) if ok else None,
                effort=float(dt*np.sum(data['effort'][j:k])) if ok else None))
        out['regimes'].append({'segment':q,'regime':str(data['labels'][a]),
            'cost_with_failure_penalty':float(dt*costs[a:b].sum()),
            'position_rmse_observed':float(np.sqrt(np.nanmean(err[a:b]**2))) if np.isfinite(err[a:b]).any() else None})
    for j,name in enumerate(['action','estimation','update','total']):
        out[name+'_ms_mean']=float(np.mean(times[:,j])*1000)
        out[name+'_ms_p95']=float(np.quantile(times[:,j],.95)*1000)
        out[name+'_ms_p99']=float(np.quantile(times[:,j],.99)*1000)
    out['deadline_miss_frequency']=float(np.mean(times[:,3]>dt))
    return out
