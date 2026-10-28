"""Dashboard adapter for local Monte Carlo ensembles around a deterministic fit.

Reuses orbit_dashboard's six-element multi-panel visualization without
misrepresenting proposal draws as posterior probabilities.
"""
from types import SimpleNamespace
import numpy as np
from orbit_dashboard import plot_dashboard


def plot_fitted_dashboard(mc, times, ra_deg, dec_deg, lat_deg, lon_deg,
                          altitude_km=0., truth_state=None, truth_epoch=None,
                          output=None, max_tracks=45, max_orbit_tracks=18,
                          sky_prediction_minutes=120):
    """Plot a fitted-ensemble dashboard; uniform weights describe draws, not a posterior.

    mc: orbit_mc_fitted.LocalMCResult. All stored states already pass physical
    and tracklet-compatibility filters. Truth is optional (synthetic tests only).
    """
    n=len(mc.states)
    if n==0:
        raise ValueError('No accepted Monte Carlo orbits to plot')
    adapter=SimpleNamespace(
        accepted=np.ones(n,dtype=bool),
        states=mc.states,
        weights=np.full(n,1./n),
        chi2=mc.chi2,
        epoch=mc.epoch,
        n_drawn=mc.n_drawn,
        max_chi2=mc.chi2_limit,
        sigma_arcsec=None,  # Existing dashboard uses this for residual shading.
    )
    fig=plot_dashboard(adapter,times,ra_deg,dec_deg,lat_deg,lon_deg,
                       altitude_km,truth_state=truth_state,
                       truth_epoch=truth_epoch,output=None,
                       max_tracks=max_tracks,max_orbit_tracks=max_orbit_tracks,
                       sky_prediction_minutes=sky_prediction_minutes)
    fig.suptitle('Local admissible-orbit Monte Carlo | deterministic fit',fontsize=19)
    fig.text(.5,.004,'Accepted draws from a local Gaussian proposal: contours describe the sampled ensemble, NOT posterior probabilities.',
             ha='center',fontsize=10,color='firebrick')
    if output is not None:
        fig.savefig(output,dpi=135,bbox_inches='tight')
    return fig
