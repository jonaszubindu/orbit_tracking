"""Diagnostic plots for weighted Stage B samples; display ESS prominently."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orbit_fit_multipass import elements
from orbit_toy import station_state

def plot_stage_b_dashboard(result,output):
    w=result.weights;states=result.states
    st,_=station_state(result.epoch,*result.station)
    rho=np.linalg.norm(states[:,:3]-st,axis=1)
    finite=np.flatnonzero(w>0)
    if not len(finite):raise ValueError('No weighted states')
    selected=finite[np.argsort(w[finite])[-min(1000,len(finite)):]]
    fig,ax=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    a=ax.ravel()
    a[0].scatter(rho[selected],result.chi2[selected],c=w[selected],s=8,cmap='viridis')
    a[0].set_xscale('log');a[0].set_xlabel('Topocentric range [km]');a[0].set_ylabel('Angular χ²')
    a[0].set_title('Highest-weight proposals (color = weight)')
    a[1].hist(np.log10(np.maximum(w[finite],1e-300)),bins=40)
    a[1].set_xlabel('log10 normalized importance weight');a[1].set_ylabel('Count')
    a[1].set_title('Weight degeneracy')
    a[2].hist(rho[finite],bins=45,weights=w[finite]);a[2].set_xlabel('Range [km]');a[2].set_ylabel('Weighted mass per bin')
    a[2].set_title('Range marginal (prior-dependent)')
    vals=[];ww=[]
    for j in selected:
        try:
            el=elements(states[j]);vals.append(el['a_km']);ww.append(w[j])
        except (ValueError,ZeroDivisionError):pass
    if vals:
        a[3].hist(vals,bins=40,weights=ww)
    a[3].set_xlabel('Semimajor axis [km]');a[3].set_ylabel('Weighted mass per bin')
    a[3].set_title('Semimajor axis (top weighted proposals)')
    fig.suptitle(f'Stage B | ESS {result.ess:.1f}/{result.n_draws} | top-10 weight {result.diagnostics["top10_mass"]:.1%}\n'
                 'Unvalidated global coverage; NOT calibrated mode probabilities',fontsize=12)
    fig.savefig(output,dpi=145);plt.close(fig)
