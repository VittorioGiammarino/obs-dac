"""Seed-paired bootstrap, explicit failure counts, saved-results plotting."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLES={
 'nominal_nonlinear':('Nominal PD','#718096'),
 'pid_nonlinear':('PID','#b08968'),
 'bias_linear':('Bias linear','#f6ad55'),
 'bias_nonlinear':('Bias nonlinear','#dd6b20'),
 'dob_nonlinear':('DOB','#009e73'),
 'l1_nonlinear':('L1 + PID','#cc79a7'),
 'nmpc_nonlinear':('NMPC + DOB','#9467bd'),
 'dac_linear':('DAC linear','#56b4e9'),
 'dac_nonlinear':('DAC nonlinear','#0072b2'),
}

def interval(values,rng,B=2000):
    x=np.asarray(values)
    if len(x)<2: return [None,None]
    means=x[rng.integers(len(x),size=(B,len(x)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()

def headline_bars(rows, root, variant):
    """Three comparable metrics; incomplete trajectories never masquerade as cheap runs."""
    order=list(STYLES)
    labels=[STYLES[k][0] for k in order]
    colors=[STYLES[k][1] for k in order]
    groups={key: {r['seed']: r for r in rows if r['method']+'_'+r['model']==key} for key in order}
    present=[key for key in order if groups[key]]
    common=set.intersection(*(set(groups[key]) for key in present))
    complete={seed for seed in common if all(not groups[key][seed]['failed'] for key in present)}
    specs=[('position_rmse_observed','Position RMSE','m',complete),
           ('cost_with_failure_penalty','Cumulative cost','cost × s',common),
           ('effort_observed','Control-energy proxy','N² s; deviation from hover',complete)]
    rng=np.random.default_rng(7319); table=[]
    fig,axs=plt.subplots(1,3,figsize=(18,6.5))
    for ax,(metric,title,unit,seeds) in zip(axs,specs):
        for j,key in enumerate(present):
            vals=[groups[key][seed][metric] for seed in sorted(seeds)]
            mean=float(np.mean(vals)) if vals else None
            ci=interval(vals,rng) if vals else [None,None]
            table.append(dict(variant=variant,controller=key,metric=metric,mean=mean,
                              ci95_low=ci[0],ci95_high=ci[1],n=len(vals),
                              failures=sum(r['failed'] for r in groups[key].values()),
                              available_seeds=len(groups[key]),seeds=sorted(seeds)))
            if vals:
                ax.bar(j,mean,color=colors[order.index(key)],width=.72)
                if ci[0] is not None:
                    ax.vlines(j,ci[0],ci[1],color='black',linewidth=1.3)
                    ax.hlines(ci,[j-.08,j-.08],[j+.08,j+.08],color='black',linewidth=1.3)
            else:
                ax.text(j,.03,'N/A',ha='center',transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(present)),[labels[order.index(key)] for key in present],rotation=35,ha='right')
        ax.set_title(title+'\n'+f'n = {len(seeds)} paired seeds',fontsize=12)
        ax.set_ylabel(unit); ax.set_ylim(bottom=0); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    failures='; '.join(labels[order.index(key)].replace('\n',' ')+': '+str(sum(r['failed'] for r in groups[key].values()))+'/'+str(len(groups[key])) for key in present)
    note=('Bars: seed means; whiskers: 95% seed-bootstrap intervals (omitted for n < 2).\n'
          'Cost: all paired seeds, including failure-tail penalties. RMSE and effort: seeds completed by every controller.\n'
          'Failures: '+failures)
    note=note.replace('; L1', ';\nL1')
    if len(common)==1: note+='\nSingle-seed execution check only; not research evidence.'
    fig.suptitle(variant+' — controller comparison',fontsize=14)
    fig.text(.02,.02,note,fontsize=8,va='bottom')
    fig.tight_layout(rect=(0,.25,1,.93))
    for ext in ['png','pdf']: fig.savefig(root/f'{variant}_headline_metrics.{ext}',dpi=180)
    plt.close(fig)
    (root/f'{variant}_headline_metrics.json').write_text(json.dumps(table,indent=2))
    with (root/f'{variant}_headline_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)

def save_table(root,name,rows):
    (root/f'{name}.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (root/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def trace_plots(root,variant,rr):
    seed=min(r['seed'] for r in rr)
    cfg=json.loads((root/variant/'config.json').read_text()); dt=cfg['dt']
    loaded={}
    for r in rr:
        if r['seed']!=seed: continue
        key=r['method']+'_'+r['model']; f=root/variant/f'{seed}_{key}.npz'
        if f.exists():
            with np.load(f) as z: loaded[key]={k:z[k].copy() for k in z.files}
    if not loaded: return
    fig,axs=plt.subplots(5,1,figsize=(12,12),sharex=True)
    diag,ds=plt.subplots(5,1,figsize=(12,12),sharex=True)
    traj,ta=plt.subplots(figsize=(8,7))
    for key,z in loaded.items():
        label,color=STYLES[key]; t=np.arange(len(z['x']))*dt
        axs[0].plot(t,z['x'][:,0],color=color,label=label)
        axs[1].plot(t,z['x'][:,1],color=color)
        axs[2].plot(t,np.linalg.norm(z['x'][:,:2]-z['ref'][:,:2],axis=1),color=color)
        axs[3].plot(t,z['x'][:,4],color=color)
        for j,style in enumerate(['-','--']):
            axs[4].plot(t,z['u'][:,j],style,color=color,alpha=.8)
            if key!='nmpc_nonlinear': ds[0].plot(t,z['v'][:,j],style,color=color,alpha=.8,label=label if j==0 else None)
            ds[1].plot(t,z['z'][:,j],style,color=color,alpha=.7)
        if 'extra_input' in z:
            ds[2].plot(t,z['extra_input'][:,0],color=color)
            ds[2].plot(t,z['extra_input'][:,1],'--',color=color)
        if key.startswith('dac'):
            ds[3].plot(t,z['grad_norm'],color=color,label=label+' gradient')
            ds[3].plot(t,z['parameter_norm'],'--',color=color,label=label+' parameters')
        ds[4].plot(t,1000*z['timing'][:,3],color=color)
        ta.plot(z['x'][:,0],z['x'][:,1],color=color,label=label)
    for j in [0,1]: axs[j].plot(t,z['ref'][:,j],'k:',label='reference' if j==0 else None)
    ta.plot(z['ref'][:,0],z['ref'][:,1],'k:',label='reference')
    for axes in [axs,ds]:
        for ax in axes:
            for b in np.arange(cfg['scenario']['duration'],len(t)*dt,cfg['scenario']['duration']): ax.axvline(b,color='grey',alpha=.2)
            ax.grid(alpha=.2)
    for ax,label in zip(axs,['x (m)','z (m)','position error (m)','pitch (rad)','rotors (N), L solid / R dashed']): ax.set_ylabel(label)
    for ax,label in zip(ds,['outer correction (m/s²)','raw feature (m/s²)','extra thrust (N) / torque (Nm)','gradient / parameter norm','controller time (ms)']): ax.set_ylabel(label)
    axs[0].legend(ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.5))
    ds[0].legend(ncol=4,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.5))
    if ds[3].get_legend_handles_labels()[0]: ds[3].legend(fontsize=8)
    axs[-1].set_xlabel('time (s)'); ds[-1].set_xlabel('time (s)')
    fig.suptitle(f'{variant}: representative seed {seed}',y=.998)
    diag.suptitle('Diagnostics: consistent controller colors; x solid / z dashed. NMPC has no outer correction.',y=.998,fontsize=10)
    for f,name in [(fig,'timeseries'),(diag,'diagnostics')]:
        f.tight_layout(rect=(0,0,1,.96)); f.savefig(root/f'{variant}_{name}.png',dpi=150); plt.close(f)
    ta.set(xlabel='horizontal position (m)',ylabel='altitude (m)'); ta.legend(fontsize=8)
    traj.tight_layout(); traj.savefig(root/f'{variant}_trajectory.png',dpi=150); plt.close(traj)


def main():
    p=argparse.ArgumentParser(); p.add_argument('results'); a=p.parse_args(); root=Path(a.results)
    rows=[]
    for f in sorted(root.glob('*/*.json')):
        obj=json.loads(f.read_text())
        if isinstance(obj,dict) and all(k in obj for k in ['seed','method','model','variant']): rows.append(obj)
    if not rows: raise SystemExit('No run metrics found')
    unknown={r['method']+'_'+r['model'] for r in rows}-set(STYLES)
    for key in sorted(unknown): STYLES[key]=(key,'#444444')
    rng=np.random.default_rng(9181); summary=[]; paired=[]; regimes=[]; phases=[]
    for variant in dict.fromkeys(r['variant'] for r in rows):
        rr=[r for r in rows if r['variant']==variant]; headline_bars(rr,root,variant)
        groups={key:[r for r in rr if r['method']+'_'+r['model']==key] for key in STYLES}
        groups={k:v for k,v in groups.items() if v}
        for key,gg in groups.items():
            costs=[r['cost_with_failure_penalty'] for r in gg]
            metrics={}
            names=['position_rmse_observed','velocity_rmse_observed','effort_observed','saturation_frequency_observed',
                   'max_abs_pitch','position_cost_observed','velocity_cost_observed','effort_cost_observed',
                   'action_ms_mean','estimation_ms_mean','update_ms_mean','total_ms_mean','total_ms_p95',
                   'total_ms_p99','deadline_miss_frequency','solver_nonconvergence_frequency']
            for metric in names:
                vals=[r[metric] for r in gg if metric in r and r[metric] is not None]
                if vals: metrics[metric]={'mean':float(np.mean(vals)),'ci95':interval(vals,rng)}
            summary.append(dict(variant=variant,controller=key,n=len(gg),failures=sum(r['failed'] for r in gg),
                mean_cost=float(np.mean(costs)),ci95=interval(costs,rng),metrics=metrics,
                mean_controller_ms=float(np.mean([r['total_ms_mean'] for r in gg]))))
            regime_names=list(dict.fromkeys(q['regime'] for r in gg for q in r['regimes']))
            for regime in regime_names:
                vals=[np.mean([q['cost_with_failure_penalty'] for q in r['regimes'] if q['regime']==regime]) for r in gg
                      if any(q['regime']==regime for q in r['regimes'])]
                regimes.append(dict(variant=variant,controller=key,regime=regime,n=len(vals),mean_cost=float(np.mean(vals)),ci95=interval(vals,rng)))
                for phase in ['transition','later']:
                    vals=[np.mean([q['mean_stage_cost'] for q in r.get('phases',[]) if q['regime']==regime and q['phase']==phase]) for r in gg
                          if any(q['regime']==regime and q['phase']==phase for q in r.get('phases',[]))]
                    if vals: phases.append(dict(variant=variant,controller=key,regime=regime,phase=phase,n=len(vals),mean_stage_cost=float(np.mean(vals)),ci95=interval(vals,rng)))
        pairs=[(left,right) for left in groups if left.startswith('dac_') for right in groups if right!=left and not right.startswith('dac_')]
        if 'dac_linear' in groups and 'dac_nonlinear' in groups: pairs.append(('dac_nonlinear','dac_linear'))
        for left,right in pairs:
            ll={r['seed']:r for r in groups[left]}; rrmap={r['seed']:r for r in groups[right]}
            common=sorted(ll.keys()&rrmap.keys())
            if any(ll[s]['scenario_hash']!=rrmap[s]['scenario_hash'] for s in common): raise ValueError('Unpaired scenarios')
            for metric in ['cost_with_failure_penalty','position_rmse_observed','effort_observed']:
                seeds=common if metric=='cost_with_failure_penalty' else [s for s in common if not ll[s]['failed'] and not rrmap[s]['failed']]
                vals=[ll[s][metric]-rrmap[s][metric] for s in seeds]
                paired.append(dict(variant=variant,left=left,right=right,metric=metric,n=len(seeds),
                                   mean_difference=float(np.mean(vals)) if vals else None,ci95=interval(vals,rng) if vals else [None,None]))
        trace_plots(root,variant,[r for g in groups.values() for r in g])
        rg=[r for r in regimes if r['variant']==variant]; fig,ax=plt.subplots(figsize=(10,5))
        for key in groups:
            gg=[r for r in rg if r['controller']==key]; label,color=STYLES[key]
            ax.plot([r['regime'] for r in gg],[r['mean_cost'] for r in gg],'o-',label=label,color=color)
        ax.set(ylabel='mean segment cost',title=variant+' — per-regime costs (history carries across switches)')
        ax.legend(fontsize=8,ncol=3); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(root/f'{variant}_regime_comparison.png',dpi=150); plt.close(fig)
        pp=[r for r in phases if r['variant']==variant]
        if pp:
            fig,axs=plt.subplots(1,2,figsize=(13,5))
            for ax,phase in zip(axs,['transition','later']):
                for key in groups:
                    gg=[r for r in pp if r['controller']==key and r['phase']==phase]; label,color=STYLES[key]
                    if gg: ax.plot([r['regime'] for r in gg],[r['mean_stage_cost'] for r in gg],'o-',label=label,color=color)
                ax.set(title=phase,ylabel='mean stage cost'); ax.grid(alpha=.2)
            axs[0].legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(root/f'{variant}_transition_comparison.png',dpi=150); plt.close(fig)
    for name,table in [('summary',summary),('paired',paired),('regimes',regimes),('phases',phases)]: save_table(root,name,table)
    fig,ax=plt.subplots(figsize=(11,6))
    for key,(label,color) in STYLES.items():
        ss=[s for s in summary if s['controller']==key]
        if ss: ax.scatter([s['mean_controller_ms'] for s in ss],[s['mean_cost'] for s in ss],color=color,label=label)
    ax.set(xlabel='mean controller time (ms; log scale)',ylabel='cost including failure penalties',xscale='log')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(root/'computation.png',dpi=150); plt.close(fig)
    # Plot every configured variant, including model-error and disturbance stress cases.
    fig,ax=plt.subplots(figsize=(max(10,len({r['variant'] for r in rows})*.6),6))
    for key,(label,color) in STYLES.items():
        ss=[s for s in summary if s['controller']==key]
        if ss: ax.plot([s['variant'] for s in ss],[s['mean_cost'] for s in ss],'o-',label=label,color=color)
    ax.tick_params(axis='x',rotation=45); ax.set_ylabel('cost including failure penalties'); ax.legend(ncol=3,fontsize=8)
    fig.tight_layout(); fig.savefig(root/'variants_comparison.png',dpi=150); plt.close(fig)

if __name__=='__main__': main()
