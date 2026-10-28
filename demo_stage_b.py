"""Run Stage B importance-sampling posterior on an existing Stage A NPZ."""
import argparse
from pathlib import Path
import numpy as np
from orbit_stage_b import run_stage_b
from orbit_stage_b_dashboard import plot_stage_b_dashboard

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-a',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('results/stage_b'))
    p.add_argument('--draws',type=int,default=30000)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--local-fraction',type=float,default=.90)
    p.add_argument('--angular-prior-sigma',type=float,default=12.)
    p.add_argument('--angular-proposal-sigma',type=float,default=2.)
    p.add_argument('--log-range-proposal-sigma',type=float,default=.35)
    p.add_argument('--rate-proposal-sigma',type=float,default=1.2)
    p.add_argument('--sigma-arcsec',type=float,default=5.)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    res=run_stage_b(args.stage_a,n_draws=args.draws,seed=args.seed,
        sigma_arcsec=args.sigma_arcsec,local_fraction=args.local_fraction,
        angular_prior_sigma=args.angular_prior_sigma,
        angular_proposal_sigma=args.angular_proposal_sigma,
        log_range_proposal_sigma=args.log_range_proposal_sigma,
        rate_proposal_sigma=args.rate_proposal_sigma)
    stem=args.stage_a.stem.replace('stage_a_','stage_b_')
    path=args.output_dir/f'{stem}.npz'
    np.savez_compressed(path,epoch=res.epoch,station=np.asarray(res.station),states=res.states,weights=res.weights,
        chi2=res.chi2,log_prior=res.log_prior,log_proposal=res.log_proposal,
        family_ids=res.family_ids,ess=res.ess,n_draws=res.n_draws,
        n_physical=res.n_physical,log_evidence_unnormalized=res.log_evidence_estimate)
    fig=args.output_dir/f'{stem}_dashboard.png';plot_stage_b_dashboard(res,fig)
    print(f'Wrote {path} and {fig}')
    print(f'ESS {res.ess:.2f} / {res.n_draws} ({res.ess/res.n_draws:.3%}), max weight {res.diagnostics["max_weight"]:.3f}')
    print(f'Top 10 samples hold {res.diagnostics["top10_mass"]:.2%} posterior weight')
    if res.ess<100 or res.ess/res.n_draws<.01:
        print('WARNING: Severe importance-weight degeneracy. Do NOT interpret family probabilities or sky probability maps.')
    print('Prior-dependent approximate posterior; Stage A seed coverage remains unverified.')
if __name__=='__main__':main()
