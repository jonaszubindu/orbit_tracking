"""Optional diagnostic; does not overwrite Stage A/B or legacy outputs."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orbit_connected_regions import identify_regions
from orbit_global_dashboard import load_ensemble,plot_global_dashboard

p=argparse.ArgumentParser()
p.add_argument('--stage-a',required=True)
p.add_argument('--stage-b',required=True)
p.add_argument('--output-dir',default='results/connected')
p.add_argument('--delta-chi2',type=float,default=16.812)
p.add_argument('--min-coverage',type=float,default=.8)
p.add_argument('--prediction-minutes',type=float,default=120)
p.add_argument('--max-tracks-per-family',type=int,default=16)
args=p.parse_args()
out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
with np.load(args.stage_a) as a, np.load(args.stage_b) as b:
    result=identify_regions(a,b,args.delta_chi2,args.min_coverage)
    grid=np.asarray(a['grid_chi2']);ranges=np.asarray(a['ranges_km']);rates=np.asarray(a['rates_km_s'])
    original=np.asarray(b['family_ids'],int)
    np.savez_compressed(out/'connected_region_diagnostic.npz',grid_labels=result.grid_labels,
        candidate_labels=result.labels,original_labels=original,passed=result.passed,
        assigned_weight=result.assigned_weight,threshold=result.threshold,reason=result.reason)
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
image=ax.pcolormesh(ranges,rates,result.grid_labels.T,shading='nearest',cmap='tab20')
ax.set(xscale='log',xlabel='Topocentric range (km)',ylabel='Topocentric range rate (km/s)',
       title='Coarse fixed-attributable connected regions (NOT profiled likelihood)')
fig.colorbar(image,ax=ax,label='Connected component ID');fig.savefig(out/'connected_regions.png',dpi=150);plt.close(fig)
print(result.reason)
ensemble=load_ensemble(args.stage_a,args.stage_b)
if result.passed:
    # Unassigned candidates are shown as a separate "unresolved" group, never discarded.
    ensemble.family_ids=result.labels
    print('Using connected-grid labels; label 0 = unassigned / unresolved')
else:
    print('Using unchanged original Stage B family labels')
info=plot_global_dashboard(ensemble,out/'global_dashboard.png',
    prediction_minutes=args.prediction_minutes,max_tracks_per_family=args.max_tracks_per_family)
print('Dashboard:',info['output'])
