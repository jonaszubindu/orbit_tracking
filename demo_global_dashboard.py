"""Independent dashboard CLI. No changes to original Monte Carlo workflow."""
import argparse
from orbit_global_dashboard import load_ensemble,plot_global_dashboard


def main():
    p=argparse.ArgumentParser(description='Visualize Stage A+B orbit families without modifying the original dashboard')
    p.add_argument('--stage-a',required=True,help='Stage A .npz file')
    p.add_argument('--stage-b',required=True,help='Stage B .npz file')
    p.add_argument('--legacy-mc',default=None,help='Optional legacy local MC .npz, diagnostic only')
    p.add_argument('--output',default='results/global_family_dashboard.png')
    p.add_argument('--prediction-minutes',type=float,default=120)
    p.add_argument('--max-tracks-per-family',type=int,default=16)
    args=p.parse_args()
    data=load_ensemble(args.stage_a,args.stage_b,args.legacy_mc)
    summary=plot_global_dashboard(data,args.output,args.prediction_minutes,args.max_tracks_per_family)
    print('Saved:',summary['output']);print('ESS:',summary['ess']);print('Provisional family weights:',summary['family_mass'])

if __name__=='__main__':main()
