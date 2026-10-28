"""Compare local Monte Carlo ensembles for 3-, 5-, 10-, 15-minute and two-pass arcs.

Usage:
    python demo_mc_fitted.py --quick
    python demo_mc_fitted.py --draws 2000 --dashboard --dashboard-tracks 30

The two original long-arc noise realizations are preserved. The 3-, 5- and
10-minute cases use independent, reproducible noise streams.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from demo_multipass_fit import LAT, LON, ALT, T0, SIGMA, make_truth, generate
from orbit_fit_multipass import fit_orbit, propagate_at, elements
from orbit_mc_fitted import sample_near_fit

CASES = {
    '3 min': np.linspace(0, 180, 20),
    '5 min': np.linspace(0, 300, 30),
    '10 min': np.linspace(0, 600, 45),
    '15 min': np.linspace(0, 900, 60),
    'two passes': np.r_[np.linspace(0, 900, 60), np.linspace(7200, 8100, 60)],
}

def make_case_noise():
    # Keep the previously published 15-minute and two-pass datasets identical.
    old = np.random.default_rng(3)
    old.normal(size=(8, 2))  # Historical 70-second case.
    return {
        '3 min': np.random.default_rng(303).normal(size=(20, 2)),
        '5 min': np.random.default_rng(305).normal(size=(30, 2)),
        '10 min': np.random.default_rng(310).normal(size=(45, 2)),
        '15 min': old.normal(size=(60, 2)),
        'two passes': old.normal(size=(120, 2)),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--draws', type=int, default=600)
    p.add_argument('--quick', action='store_true', help='150 draws per case')
    p.add_argument('--dashboard', action='store_true')
    p.add_argument('--dashboard-tracks', type=int, default=45)
    p.add_argument('--dashboard-orbit-tracks', type=int, default=18)
    p.add_argument('--sky-prediction-minutes', type=float, default=120.)
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth = make_truth()
    noise = make_case_noise()
    fig, axes = plt.subplots(len(CASES), 3, figsize=(14, 3.1 * len(CASES)), layout='constrained')
    summary = []
    for row, (name, seconds) in enumerate(CASES.items()):
        times = T0 + np.rint(seconds).astype('timedelta64[s]')
        ra, dec = generate(times, truth, noise[name])
        case = {'case': name, 'n_observations': len(times),
                'span_s': float(seconds[-1] - seconds[0]),
                'fit_success': False, 'fit_chi2': np.nan,
                'chi2_limit': np.nan, 'n_drawn': 0,
                'n_physical': 0, 'n_accepted': 0, 'acceptance_rate': np.nan,
                'error': ''}
        try:
            fit = fit_orbit(times, ra, dec, LAT, LON, ALT,
                            circular_ranges=7, circular_phases=4,
                            max_nfev=120, extra_seeds=4)
            case['fit_success'] = bool(fit.success)
            case['fit_chi2'] = float(fit.fitted_chi2)
            case['circular_chi2'] = float(fit.circular_chi2)
            truth_epoch = propagate_at(truth, [float((fit.epoch-T0)/np.timedelta64(1, 's'))])[0]
            true_el = elements(truth_epoch)
            fit_el = elements(fit.fitted_state)
            case.update({f'true_{k}': v for k, v in true_el.items()})
            case.update({f'fit_{k}': v for k, v in fit_el.items()})
            if not fit.success:
                raise RuntimeError('Deterministic fit did not converge to a physical free-orbit solution')
            mc = sample_near_fit(fit, times, ra, dec, LAT, LON, ALT,
                                 sigma_arcsec=SIGMA,
                                 n_draws=150 if args.quick else args.draws,
                                 seed=123 + row)
            case.update(n_drawn=mc.n_drawn, n_physical=mc.n_physical,
                        n_accepted=len(mc.states), acceptance_rate=mc.acceptance_rate,
                        chi2_limit=mc.chi2_limit)
            print(f'{name}: fit χ²={mc.reference_chi2:.2f}, '
                  f'accepted={len(mc.states)}/{mc.n_drawn}, physical={mc.n_physical}', flush=True)
            if len(mc.states):
                for ax, key in zip(axes[row], ('a_km', 'e', 'i_deg')):
                    ax.hist(mc.elements[key], bins=25, color='#5375a8', alpha=.75,
                            label='Accepted draws')
                    ax.axvline(true_el[key], color='#ca533f', ls='--', lw=2, label='Truth')
                    ax.axvline(fit_el[key], color='#252525', lw=1.5, label='Fit')
                    ax.set_xlabel(key)
                    ax.set_ylabel('Count')
                    ax.grid(alpha=.2)
                    if ax is axes[row, 0]:
                        ax.legend(fontsize=8)
                    ax.set_title(f'{name} | {key}')
                if args.dashboard:
                    from orbit_fitted_dashboard import plot_fitted_dashboard
                    output = args.output_dir / f'mc_fitted_dashboard_{name.replace(" ", "_")}.png'
                    dashboard = plot_fitted_dashboard(mc, times, ra, dec, LAT, LON, ALT,
                        truth_state=truth, truth_epoch=T0, output=output,
                        max_tracks=args.dashboard_tracks,
                        max_orbit_tracks=args.dashboard_orbit_tracks,
                        sky_prediction_minutes=args.sky_prediction_minutes)
                    plt.close(dashboard)
                    print(f'  Dashboard: {output}', flush=True)
            else:
                for ax in axes[row]:
                    ax.text(.5, .5, 'No accepted orbits', transform=ax.transAxes,
                            ha='center', va='center')
            np.savez_compressed(args.output_dir / f'mc_fitted_{name.replace(" ", "_")}.npz',
                states=mc.states, chi2=mc.chi2, epoch=str(mc.epoch),
                reference_state=mc.reference_state, chi2_limit=mc.chi2_limit,
                **mc.elements)
        except Exception as exc:
            case['error'] = str(exc)
            print(f'{name}: SKIPPED Monte Carlo: {exc}', flush=True)
            for ax in axes[row]:
                ax.text(.5, .5, f'No valid ensemble\n{exc}', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, wrap=True)
                ax.set_title(name)
        summary.append(case)
    fig.suptitle('Local admissible-orbit ensembles: 3, 5, 10, 15 minutes and two passes\n'
                 'Proposal distributions, not calibrated posteriors')
    comparison = args.output_dir / 'mc_fitted_comparison.png'
    fig.savefig(comparison, dpi=150)
    plt.close(fig)
    summary_path = args.output_dir / 'mc_fitted_summary.csv'
    fields = list(dict.fromkeys(k for case in summary for k in case))
    with summary_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f'Comparison: {comparison}\nSummary: {summary_path}', flush=True)
    return summary

if __name__ == '__main__':
    main()
