"""Repeated-noise, fixed-truth predictive spherical-cap coverage diagnostic.

NOT simulation-based calibration (SBC): truth is fixed, not drawn from the
Stage B prior. NOT HPD sky regions: regions are nested spherical caps around
weighted medoid. Low ESS and failure rates are always reported.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from orbit_toy import station_state, propagate
from orbit_fit_multipass import propagate_at
from orbit_stage_a import search_stage_a
from orbit_stage_b import run_stage_b
from demo_multipass_fit import T0,LAT,LON,ALT,SIGMA,make_truth,generate

LEVELS=(0.5,0.9,0.95)

def directions(states,epoch,t,station):
    dt=float((t-epoch)/np.timedelta64(1,'s'))
    r,_=propagate(states[:,:3].copy(),states[:,3:].copy(),dt,max_step=30.)
    d=r-station_state(t,*station)[0]
    return d/np.linalg.norm(d,axis=1)[:,None]

def credible_caps(u,w,truth_u,levels=LEVELS):
    """Return radii and inclusion for posterior-mass spherical caps.

    Center is highest-mass sample in a 2-degree neighborhood when feasible;
    otherwise the maximum-weight sample. No claim of minimum-area/HPD region.
    """
    valid=np.isfinite(w)&(w>0)
    u=np.asarray(u)[valid];w=np.asarray(w)[valid];w=w/w.sum()
    if len(w)==0: raise ValueError('No nonzero-weight states')
    tree=cKDTree(u)
    neighbors=tree.query_ball_point(u,2*np.sin(np.deg2rad(2)/2))
    mass=np.array([w[idx].sum() for idx in neighbors])
    center=u[np.argmax(mass)]
    theta=np.arccos(np.clip(u@center,-1,1))
    order=np.argsort(theta)
    cum=np.cumsum(w[order]);radii=np.interp(levels,cum,theta[order],left=theta[order][0],right=theta[order][-1])
    # Discrete empirical quantiles use the first sample reaching requested mass
    radii=np.array([theta[order][min(np.searchsorted(cum,lev),len(cum)-1)] for lev in levels])
    true_sep=float(np.arccos(np.clip(truth_u@center,-1,1)))
    return radii, true_sep<=radii+1e-12, true_sep

def run(args):
    rng=np.random.default_rng(args.seed)
    seconds=np.linspace(0,args.arc_minutes*60,args.n_obs)
    times=T0+np.rint(seconds).astype('timedelta64[s]')
    truth=make_truth();station=(args.search_lat,args.search_lon,args.search_alt_km)
    pred_t=T0+np.timedelta64(int(round(args.prediction_hours*3600)),'s')
    true_u=directions(truth[None],T0,pred_t,station)[0]
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[];start=time.monotonic()
    for k in range(args.trials):
        row={'trial':k,'success':False,'ess':float('nan'),'elapsed_s':float('nan')}
        t0=time.monotonic()
        try:
            ra,dec=generate(times,truth,rng.normal(size=(len(times),2)))
            a=search_stage_a(times,ra,dec,LAT,LON,ALT,sigma_arcsec=SIGMA,
                n_range=args.n_range,n_rate=args.n_rate,n_refine=args.n_refine,max_nfev=args.max_nfev)
            apath=out/f'_stage_a_trial_{k}.npz'
            np.savez_compressed(apath,epoch=a.epoch,states=a.states,chi2=a.chi2,family_ids=a.family_ids,
                times=times,ra_deg=ra,dec_deg=dec,station_lat_deg=LAT,station_lon_deg=LON,station_alt_km=ALT)
            b=run_stage_b(apath,n_draws=args.draws,seed=args.seed+k+1000,batch_size=400,max_step_s=30.)
            if not args.keep_intermediates: apath.unlink()
            u=directions(b.states,b.epoch,pred_t,station)
            radii,covered,sep=credible_caps(u,b.weights,true_u)
            row.update(success=True,ess=b.ess,n_nonzero=int(np.count_nonzero(b.weights)),
                true_separation_deg=float(np.rad2deg(sep)),
                **{f'radius_{int(p*100)}_deg':float(np.rad2deg(r)) for p,r in zip(LEVELS,radii)},
                **{f'covered_{int(p*100)}':bool(c) for p,c in zip(LEVELS,covered)})
        except Exception as e:
            row['error']=f'{type(e).__name__}: {e}'
        row['elapsed_s']=round(time.monotonic()-t0,2)
        rows.append(row)
        print(f"trial {k+1}/{args.trials}: success={row['success']} ESS={row['ess']:.1f} elapsed={row['elapsed_s']:.1f}s",flush=True)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with (out/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    ok=[r for r in rows if r['success']]
    summary={'experiment':'fixed-truth repeated-noise predictive spherical-cap coverage; NOT SBC',
        'n_trials':args.trials,'n_success':len(ok),'n_failure':args.trials-len(ok),
        'ess_median':float(np.median([r['ess'] for r in ok])) if ok else None,
        'prediction_hours':args.prediction_hours,'arc_minutes':args.arc_minutes,
        'n_draws':args.draws,'runtime_s':round(time.monotonic()-start,1),
        'coverage_conditional_on_success':{str(int(p*100)):float(np.mean([r[f'covered_{int(p*100)}'] for r in ok])) if ok else None for p in LEVELS},
        'coverage_failure_as_miss':{str(int(p*100)):float(sum(r[f'covered_{int(p*100)}'] for r in ok)/args.trials) for p in LEVELS},
        'caveat':'Only calibrated if truth is sampled from declared prior and inference is reliable; this run uses one fixed orbit.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    labels=['50%','90%','95%']; nominal=np.array(LEVELS)
    empirical=np.array([summary['coverage_failure_as_miss'][str(int(p*100))] for p in LEVELS])
    axes[0].plot([0,1],[0,1],ls='--',color='gray',label='Ideal calibration')
    axes[0].plot(nominal,empirical,'o-',label='Observed (failures count as misses)')
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Nominal cap mass',ylabel='Truth inclusion frequency',title=f'Coverage diagnostic (N={args.trials})')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.3)
    if ok: axes[1].hist([r['ess'] for r in ok],bins=min(10,max(1,len(ok))),label='Successful trial ESS')
    axes[1].set(xlabel='Effective sample size',ylabel='Number of trials',title='Importance-sampling reliability')
    axes[1].legend(fontsize=8) if ok else None
    fig.suptitle('FIXED-TRUTH pilot: not prior-predictive SBC; low N is inconclusive')
    fig.savefig(out/'coverage_diagnostic.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials',type=int,default=3);p.add_argument('--draws',type=int,default=500)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--arc-minutes',type=float,default=5)
    p.add_argument('--n-obs',type=int,default=30);p.add_argument('--prediction-hours',type=float,default=4)
    p.add_argument('--n-range',type=int,default=14);p.add_argument('--n-rate',type=int,default=17)
    p.add_argument('--n-refine',type=int,default=7);p.add_argument('--max-nfev',type=int,default=18)
    p.add_argument('--search-lat',type=float,default=-29.2567)
    p.add_argument('--search-lon',type=float,default=-70.7346)
    p.add_argument('--search-alt-km',type=float,default=2.4)
    p.add_argument('--keep-intermediates',action='store_true')
    p.add_argument('--output-dir',default='calibration/results/pilot')
    run(p.parse_args())
