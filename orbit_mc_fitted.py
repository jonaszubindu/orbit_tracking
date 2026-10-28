"""Local Monte Carlo ensemble around a successful deterministic angles-only orbit fit.

No MCMC. Samples a regularized, local linearized likelihood ellipsoid and
accepts only bound, above-minimum-perigee states satisfying a chi-square rule.
The draws are *proposal samples*, not an exact Bayesian posterior.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import chi2 as chi2_distribution
from orbit_fit_multipass import propagate_at, elements
from orbit_toy import ARCSEC, orbital_filter, station_state

SCALES = np.array([1000.,1000.,1000.,1.,1.,1.])

@dataclass
class LocalMCResult:
    epoch: np.datetime64
    states: np.ndarray
    chi2: np.ndarray
    elements: dict
    reference_state: np.ndarray
    reference_chi2: float
    chi2_limit: float
    proposal_covariance: np.ndarray
    jacobian_singular_values: np.ndarray
    n_drawn: int
    n_physical: int
    acceptance_rate: float


def sample_near_fit(fit, times, ra_deg, dec_deg, lat_deg, lon_deg,
                    altitude_km=0., sigma_arcsec=5., n_draws=600,
                    confidence=0.95, proposal_scale=1., seed=42,
                    min_perigee_alt_km=100., relative_singular_floor=1e-7,
                    max_scaled_std=20., batch_size=100):
    """Return physically admissible candidate states near the fitted solution.

    Threshold: chi2(state) <= chi2(best_fit) + chi2.ppf(confidence, df=6).
    This is a *local likelihood-ratio region*, not a calibrated frequentist
    confidence region for weakly identified or boundary-constrained orbits.
    The inverse-Jacobian covariance is regularized and clipped, explicitly
    defining the proposal support in nearly unconstrained directions.
    """
    if not fit.success:
        raise ValueError('Fit did not converge successfully; only use recovered fits for now.')
    if n_draws < 1 or not 0 < confidence < 1 or sigma_arcsec <= 0:
        raise ValueError('Invalid sampling parameters')
    t=np.asarray(times,dtype='datetime64[ns]')
    ra=np.deg2rad(np.asarray(ra_deg,float));dec=np.deg2rad(np.asarray(dec_deg,float))
    if len(t)!=len(ra) or len(t)!=len(dec):
        raise ValueError('Observation array lengths differ')
    dt=np.asarray((t-fit.epoch)/np.timedelta64(1,'s'),float)
    stations=np.array([station_state(ti,lat_deg,lon_deg,altitude_km)[0] for ti in t])
    center=np.asarray(fit.fitted_state,float)
    def residual_scaled(x):
        s=center+x*SCALES
        xyz=propagate_at(s,dt)[:,:3]-stations
        pred_ra=np.arctan2(xyz[:,1],xyz[:,0])
        pred_dec=np.arctan2(xyz[:,2],np.hypot(xyz[:,0],xyz[:,1]))
        dra=np.arctan2(np.sin(pred_ra-ra),np.cos(pred_ra-ra))
        return np.column_stack((dra,pred_dec-dec)).ravel()*ARCSEC/sigma_arcsec
    baseline=residual_scaled(np.zeros(6))
    base_chi2=float(baseline@baseline)
    # Centered finite differences in *scaled Cartesian coordinates*.
    J=approx_derivative(residual_scaled,np.zeros(6),method='3-point',
                        abs_step=np.array([1e-4]*3+[1e-4]*3))
    _,sv,vt=np.linalg.svd(J,full_matrices=False)
    floor=max(sv[0]*relative_singular_floor,1e-12)
    std=np.minimum(1/np.maximum(sv,floor),max_scaled_std)
    covariance=(vt.T*(std**2))@vt
    limit=base_chi2+float(chi2_distribution.ppf(confidence,df=6))
    rng=np.random.default_rng(seed)
    accepted=[];scores=[];n_physical=0
    for start in range(0,n_draws,batch_size):
        n=min(batch_size,n_draws-start)
        z=rng.normal(size=(n,6))
        delta=(z*std)@vt * proposal_scale
        states=center+delta*SCALES
        valid,_=orbital_filter(states[:,:3],states[:,3:],min_perigee_alt_km)
        n_physical+=int(valid.sum())
        for state in states[valid]:
            x=(state-center)/SCALES
            rr=residual_scaled(x)
            chi=float(rr@rr)
            if chi<=limit:
                accepted.append(state);scores.append(chi)
    states=np.asarray(accepted,float).reshape(-1,6)
    scores=np.asarray(scores,float)
    keys=('a_km','e','i_deg','raan_deg','argp_deg','M_deg')
    els={k:np.array([elements(s)[k] for s in states]) for k in keys}
    return LocalMCResult(fit.epoch,states,scores,els,center,base_chi2,
                         limit,covariance,sv,n_draws,n_physical,
                         len(states)/n_draws)
