"""Run global Stage A independently. B and C are separate future stages."""
import argparse
from pathlib import Path
import numpy as np
from demo_multipass_fit import LAT,LON,ALT,T0,SIGMA,make_truth,generate
from orbit_stage_a import search_stage_a
from orbit_stage_a_dashboard import plot_stage_a_dashboard
from orbit_fit_multipass import propagate_at

CASES={'3_min':(180,20,303),'5_min':(300,30,305),'10_min':(600,45,310)}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=[*CASES,'15_min','two_passes'],default='3_min')
    p.add_argument('--n-range',type=int,default=35);p.add_argument('--n-rate',type=int,default=41)
    p.add_argument('--n-refine',type=int,default=35);p.add_argument('--max-nfev',type=int,default=35)
    p.add_argument('--delta-chi2',type=float,default=16.812)
    p.add_argument('--prediction-minutes',type=float,default=120)
    p.add_argument('--output-dir',type=Path,default=Path('results/stage_a'))
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    truth=make_truth()
    if args.case in CASES:
        span,n,seed=CASES[args.case];seconds=np.linspace(0,span,n);noise=np.random.default_rng(seed).normal(size=(n,2))
    else:
        rng=np.random.default_rng(3);rng.normal(size=(8,2))
        if args.case=='15_min':seconds=np.linspace(0,900,60);noise=rng.normal(size=(60,2))
        else:
            seconds=np.r_[np.linspace(0,900,60),np.linspace(7200,8100,60)]
            rng.normal(size=(60,2));noise=rng.normal(size=(120,2))
    times=T0+np.rint(seconds).astype('timedelta64[s]');ra,dec=generate(times,truth,noise)
    print('Stage A: scanning bounded range/range-rate domain...',flush=True)
    res=search_stage_a(times,ra,dec,LAT,LON,ALT,sigma_arcsec=SIGMA,n_range=args.n_range,
        n_rate=args.n_rate,n_refine=args.n_refine,max_nfev=args.max_nfev,delta_chi2=args.delta_chi2)
    name=args.case;np.savez_compressed(args.output_dir/f'stage_a_{name}.npz',
        epoch=res.epoch,states=res.states,chi2=res.chi2,family_ids=res.family_ids,
        ranges_km=res.ranges_km,rates_km_s=res.rates_km_s,
        grid_chi2=res.grid_chi2,grid_physical=res.grid_physical,grid_family=res.grid_family,
        threshold=res.threshold,best_chi2=res.best_chi2,times=times,ra_deg=ra,dec_deg=dec,
        station_lat_deg=LAT,station_lon_deg=LON,station_alt_km=ALT)
    plot_stage_a_dashboard(res,args.output_dir/f'stage_a_{name}_dashboard.png',prediction_minutes=args.prediction_minutes)
    truth_epoch=propagate_at(truth,[(res.epoch-T0)/np.timedelta64(1,'s')])[0]
    errors=np.linalg.norm(res.states[:,:3]-truth_epoch[:3],axis=1)
    print(f'Case {name}: best chi2={res.best_chi2:.2f}, {len(res.states)} selected states, '
          f'{res.family_count} provisional families; nearest truth position error={errors.min():.1f} km')
    print('Stage A only: no posterior probabilities, no completeness guarantee.')
if __name__=='__main__':main()
