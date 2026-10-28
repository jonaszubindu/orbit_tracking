"""Propagate saved local orbit ensembles to Toulouse and La Silla.

Run after demo_mc_fitted.py, or use --input-dir to point to saved NPZs.
Example:
 python demo_station_search.py --input-dir results --cases 5_min two_passes
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orbit_station_search import STATIONS,predict_station_search,plot_station_search,save_search_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('results'))
    p.add_argument('--output-dir',type=Path,default=Path('results/station_search'))
    p.add_argument('--cases',nargs='+',default=['5_min','10_min','15_min','two_passes'])
    p.add_argument('--hours',type=float,default=24.)
    p.add_argument('--step-minutes',type=float,default=10.)
    p.add_argument('--min-elevation',type=float,default=10.)
    p.add_argument('--fov-radius',type=float,default=1.)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for case in args.cases:
        path=args.input_dir/f'mc_fitted_{case}.npz'
        if not path.exists():
            print(f'{case}: no saved ensemble at {path}; SKIPPED')
            continue
        with np.load(path) as data:
            states=data['states'];ref=data['reference_state'];epoch=np.datetime64(str(data['epoch']))
        if not len(states):
            print(f'{case}: empty ensemble; SKIPPED')
            continue
        for key,station in STATIONS.items():
            pred=predict_station_search(states,ref,epoch,station,
                hours=args.hours,step_minutes=args.step_minutes,
                min_elevation_deg=args.min_elevation,fov_radius_deg=args.fov_radius)
            base=f'{case}_{key}'
            save_search_csv(pred,args.output_dir/f'{base}.csv')
            fig=plot_station_search(pred,output=args.output_dir/f'{base}.png')
            plt.close(fig)
            flag='PASS POSSIBLE' if pred.has_pass else 'NO PASS'
            print(f'{case:11s} {station.name:29s} {flag:14s} '
                  f'max visible={pred.visible_fraction.max():.1%} '
                  f'max FoV={pred.fov_fraction.max():.1%} '
                  f'reference_pass={pred.reference_has_pass}',flush=True)
            for i in pred.best_times(2):
                print(f'  {str(pred.times[i])[:16]} UTC: '
                      f'visible={pred.visible_fraction[i]:.1%}, '
                      f'in FoV={pred.fov_fraction[i]:.1%}, '
                      f'RA={pred.reference_ra_deg[i]:.3f}°, Dec={pred.reference_dec_deg[i]:.3f}°')
if __name__=='__main__':main()
