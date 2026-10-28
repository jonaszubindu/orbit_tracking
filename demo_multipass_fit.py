"""Synthetic single-/two-pass deterministic OD recovery, no MCMC.
Run: python demo_multipass_fit.py
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orbit_toy import station_state, MU
from orbit_fit_multipass import fit_orbit,propagate_at,elements

LAT,LON,ALT=46.95,7.44,.55
T0=np.datetime64('2026-10-22T20:00:00','ns')
SIGMA=5.

def make_truth():
    st,_=station_state(T0,LAT,LON,ALT)
    up=st/np.linalg.norm(st);east=np.cross([0.,0.,1.],up);east/=np.linalg.norm(east)
    return np.r_[st+20000*up,3.5*east]

def generate(times,truth,noise):
    sec=np.asarray((times-T0)/np.timedelta64(1,'s'),float)
    pred=propagate_at(truth,sec)[:,:3]
    stations=np.array([station_state(t,LAT,LON,ALT)[0] for t in times]);d=pred-stations
    ra=np.rad2deg(np.arctan2(d[:,1],d[:,0]))%360
    dec=np.rad2deg(np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1])))
    return ra+noise[:,0]*SIGMA/3600,dec+noise[:,1]*SIGMA/3600

def main():
    rng=np.random.default_rng(3);truth=make_truth()
    # Two short observation windows separated by ~2 hours, with many measurements.
    cases={'70 s':np.linspace(0,70,8),
           '15 min':np.linspace(0,900,60),
           'two passes':np.r_[np.linspace(0,900,60),np.linspace(7200,8100,60)]}
    rows=[];fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,(name,seconds) in zip(axes,cases.items()):
        times=T0+np.rint(seconds).astype('timedelta64[s]')
        noise=rng.normal(size=(len(times),2));ra,dec=generate(times,truth,noise)
        fit=fit_orbit(times,ra,dec,LAT,LON,ALT,circular_ranges=7,circular_phases=4,
                      max_nfev=120,extra_seeds=4)
        offset=float((fit.epoch-T0)/np.timedelta64(1,'s'))
        truth_epoch=propagate_at(truth,np.array([offset]))[0]
        eltrue=elements(truth_epoch);elfit=elements(fit.fitted_state)
        st,_=station_state(fit.epoch,LAT,LON,ALT)
        rho_true=np.linalg.norm(truth_epoch[:3]-st);rho_fit=np.linalg.norm(fit.fitted_state[:3]-st)
        print(f'{name}: circular chi2={fit.circular_chi2:.2f}, fitted chi2={fit.fitted_chi2:.2f}, '
              f'true a={eltrue["a_km"]:.1f}, fitted a={elfit["a_km"]:.1f}, '
              f'true e={eltrue["e"]:.5f}, fitted e={elfit["e"]:.5f}, '
              f'rho truth={rho_true:.1f}, fit={rho_fit:.1f}, success={fit.success}',flush=True)
        rows.append(dict(case=name,n=len(times),span_s=float(seconds[-1]-seconds[0]),
            circular_chi2=fit.circular_chi2,fit_chi2=fit.fitted_chi2,
            truth_range_km=rho_true,fit_range_km=rho_fit,success=fit.success,
            **{f'truth_{k}':v for k,v in eltrue.items()},**{f'fit_{k}':v for k,v in elfit.items()}))
        keys=['a_km','e','i_deg'];
        for j,key in enumerate(keys):
            ax.scatter(j,elfit[key],marker='o',label='Fitted' if j==0 else None)
            ax.scatter(j,eltrue[key],marker='x',label='Truth' if j==0 else None)
        ax.set_xticks(range(3),['a [km]','e','i [deg]']);ax.set_title(f'{name} | χ²={fit.fitted_chi2:.1f}')
        ax.grid(alpha=.3);ax.legend(fontsize=8)
        # Different units across elements: use table instead of misleading joint scale.
        ax.cla();ax.axis('off')
        values=[[k,f'{eltrue[k]:.5g}',f'{elfit[k]:.5g}'] for k in ('a_km','e','i_deg','raan_deg','argp_deg','M_deg')]
        ax.table(cellText=values,colLabels=['Element','Truth','Fit'],loc='center',cellLoc='center').scale(1,1.5)
        ax.set_title(f'{name}\nχ² circular {fit.circular_chi2:.1f} → free {fit.fitted_chi2:.1f}')
    with open('/mnt/data/multipass_fit_results.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    fig.suptitle('Deterministic orbit recovery: circular initialization → 6D least-squares fit')
    fig.savefig('/mnt/data/multipass_fit_comparison.png',dpi=170)
    return rows
if __name__=='__main__':main()
