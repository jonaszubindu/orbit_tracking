"""Stage B: explicit-prior, multiple-importance-sampling orbit posterior.

This is a normalized *importance sampling approximation*, not an exhaustive
posterior or a validated SMC/MCMC. Report ESS and avoid interpreting unreliable
mode weights. State coordinates are (RA,Dec,RA_rate,Dec_rate,log_range,range_rate).
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from tracklet import fit_tracklet
from orbit_toy import ARCSEC, initial_state, station_state, orbital_filter, propagate

@dataclass
class StageBResult:
    epoch: np.datetime64
    station: tuple
    states: np.ndarray
    weights: np.ndarray
    chi2: np.ndarray
    log_prior: np.ndarray
    log_proposal: np.ndarray
    family_ids: np.ndarray
    ess: float
    n_draws: int
    n_physical: int
    n_finite: int
    log_evidence_estimate: float
    proposal_fraction: float
    prior_config: dict
    diagnostics: dict


def _parameters_from_states(states,epoch,station,center,scale):
    st,sv=station_state(epoch,*station)
    d=states[:,:3]-st
    rho=np.linalg.norm(d,axis=1)
    u=d/rho[:,None]
    vrel=states[:,3:]-sv
    rate=np.sum(vrel*u,axis=1)
    ra=np.arctan2(u[:,1],u[:,0]); dec=np.arctan2(u[:,2],np.hypot(u[:,0],u[:,1]))
    ud=(vrel-rate[:,None]*u)/rho[:,None]
    cosd=np.cos(dec)
    ar=(u[:,0]*ud[:,1]-u[:,1]*ud[:,0])/np.maximum(cosd*cosd,1e-12)
    dr=ud[:,2]/np.maximum(cosd,1e-12)
    a=np.column_stack((ra,dec,ar,dr))
    a[:,0]=center[0]+np.arctan2(np.sin(a[:,0]-center[0]),np.cos(a[:,0]-center[0]))
    return np.column_stack(((a-center)/scale,np.log(rho),rate))


def _states_from_parameters(z,epoch,station,center,scale):
    a=center+z[:,:4]*scale
    ra,dec,ar,dr=a.T
    u=np.column_stack((np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)))
    ua=np.column_stack((-np.cos(dec)*np.sin(ra),np.cos(dec)*np.cos(ra),np.zeros(len(z))))
    ud=np.column_stack((-np.sin(dec)*np.cos(ra),-np.sin(dec)*np.sin(ra),np.cos(dec)))
    rho=np.exp(np.clip(z[:,4],-100,100)); rate=z[:,5]
    st,sv=station_state(epoch,*station)
    return np.column_stack((st+rho[:,None]*u,sv+rate[:,None]*u+rho[:,None]*(ua*ar[:,None]+ud*dr[:,None])))


def _log_gaussian(z,means,covs):
    return logsumexp(np.stack([multivariate_normal.logpdf(z,mean=m,cov=c,allow_singular=False) for m,c in zip(means,covs)]),axis=0)-np.log(len(means))


def run_stage_b(stage_a_npz, n_draws=30000, sigma_arcsec=5., seed=2026,
                range_bounds=(300.,60000.),rate_bounds=(-10.,10.),
                angular_prior_sigma=12., min_perigee_alt_km=100.,
                local_fraction=.90, angular_proposal_sigma=2.,
                log_range_proposal_sigma=.35,rate_proposal_sigma=1.2,
                max_step_s=30.,batch_size=1500):
    """Multiple importance sampling with exact known proposal and explicit prior.

    Prior: independent N(0,angular_prior_sigma^2) on standardized attributable
    coordinates, uniform log-range and uniform range rate inside stated bounds,
    conditioned on bound orbit and perigee > threshold. Since the normalization
    of this physical truncation is unknown, the reported evidence is for the
    *unnormalized physical prior*, not a Bayes-factor-ready marginal likelihood.

    Proposal: mixture of local Gaussians around ALL Stage A selected states
    (equal mixture) and the untruncated prior base (fraction 1-local_fraction).
    Importance weights = likelihood * prior_base * physical / proposal.
    """
    if not (0 < local_fraction < 1): raise ValueError('local_fraction must be strictly between 0 and 1')
    if n_draws < 100: raise ValueError('Need at least 100 draws')
    with np.load(stage_a_npz) as d:
        states_a=d['states'];epoch=d['epoch'].astype('datetime64[ns]').item() if False else np.datetime64(d['epoch'],'ns')
        times=d['times'].astype('datetime64[ns]');ra=d['ra_deg'];dec=d['dec_deg']
        station=(float(d['station_lat_deg']),float(d['station_lon_deg']),float(d['station_alt_km']))
        stage_ids=d['family_ids']
    attr=fit_tracklet(times,ra,dec,sigma_arcsec)
    center=np.deg2rad([attr.ra_deg,attr.dec_deg,attr.ra_rate_deg_s,attr.dec_rate_deg_s])
    scale=np.sqrt(np.maximum(np.diag(attr.covariance),1e-28))
    centers=_parameters_from_states(states_a,epoch,station,center,scale)
    cov=np.diag([angular_proposal_sigma**2]*4+[log_range_proposal_sigma**2,rate_proposal_sigma**2])
    # Broad component is the base prior: 4D Gaussian and 2D bounded uniform.
    rng=np.random.default_rng(seed)
    islocal=rng.random(n_draws)<local_fraction
    z=np.empty((n_draws,6))
    nlocal=int(islocal.sum());chosen=rng.integers(len(centers),size=nlocal)
    z[islocal]=centers[chosen]+rng.normal(size=(nlocal,6))*np.sqrt(np.diag(cov))
    nbroad=n_draws-nlocal
    z[~islocal,:4]=rng.normal(size=(nbroad,4))*angular_prior_sigma
    z[~islocal,4]=rng.uniform(np.log(range_bounds[0]),np.log(range_bounds[1]),nbroad)
    z[~islocal,5]=rng.uniform(*rate_bounds,size=nbroad)
    within=(z[:,4]>=np.log(range_bounds[0]))&(z[:,4]<=np.log(range_bounds[1]))&(z[:,5]>=rate_bounds[0])&(z[:,5]<=rate_bounds[1])
    # Angular prior is normalized in standardized attributable coordinates.
    angular_lp=-2*np.log(2*np.pi)-4*np.log(angular_prior_sigma)-.5*np.sum((z[:,:4]/angular_prior_sigma)**2,axis=1)
    log_volume=np.log(np.log(range_bounds[1]/range_bounds[0]))+np.log(rate_bounds[1]-rate_bounds[0])
    lp=np.where(within,angular_lp-log_volume,-np.inf)
    local_lq=_log_gaussian(z,centers,[cov]*len(centers))
    broad_lq=lp # prior base density, zero outside support
    lq=np.logaddexp(np.log(local_fraction)+local_lq,np.log1p(-local_fraction)+broad_lq)
    states=_states_from_parameters(z,epoch,station,center,scale)
    physical,_=orbital_filter(states[:,:3],states[:,3:],min_perigee_alt_km)
    eligible=physical&within
    chi=np.full(n_draws,np.inf)
    idx=np.flatnonzero(eligible)
    offsets=np.asarray((times-epoch)/np.timedelta64(1,'s'),float)
    stations=np.array([station_state(t,*station)[0] for t in times])
    obsra=np.deg2rad(ra);obsdec=np.deg2rad(dec)
    for start in range(0,len(idx),batch_size):
        ids=idx[start:start+batch_size]
        r0=states[ids,:3];v0=states[ids,3:]
        score=np.zeros(len(ids))
        for dt,st,oa,od in zip(offsets,stations,obsra,obsdec):
            rp,_=propagate(r0.copy(),v0.copy(),float(dt),max_step=max_step_s)
            d=rp-st
            pa=np.arctan2(d[:,1],d[:,0]);pd=np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))
            da=np.arctan2(np.sin(pa-oa),np.cos(pa-oa))
            score+=((da*ARCSEC/sigma_arcsec)**2+(pd-od)**2*(ARCSEC/sigma_arcsec)**2)
        chi[ids]=score
    lw=-.5*chi+lp-lq
    normalizer=logsumexp(lw)
    if not np.isfinite(normalizer):raise RuntimeError('No finite posterior weights. Increase proposal breadth/draws or revise priors.')
    w=np.exp(lw-normalizer)
    ess=float(1/np.sum(w*w))
    # Assign nearest Stage A seed for diagnostic grouping only; NOT a mode probability.
    # Scaled 6D distances can merge/split real posterior components.
    std=np.array([angular_proposal_sigma]*4+[log_range_proposal_sigma,rate_proposal_sigma])
    fam=np.empty(n_draws,int)
    for start in range(0,n_draws,2000):
        dd=np.sum(((z[start:start+2000,None,:]-centers[None,:,:])/std)**2,axis=2)
        fam[start:start+2000]=stage_ids[np.argmin(dd,axis=1)]
    good=np.flatnonzero(w>0)
    logz=float(normalizer-np.log(n_draws))
    diagnostics={'max_weight':float(w.max()),'top10_mass':float(np.sort(w)[-10:].sum()),
       'ess_fraction':float(ess/n_draws),'finite_weight_draws':int(len(good)),
       'weight_in_local_proposals':float(w[islocal].sum()),
       'likelihood_model':'independent 5-arcsec Gaussian in RA/Dec coordinate angles',
       'family_labels':'nearest Stage A seed only; not validated disconnected posterior modes'}
    return StageBResult(epoch,station,states,w,chi,lp,lq,fam,ess,n_draws,int(physical.sum()),len(good),logz,
        local_fraction,{'range_bounds_km':range_bounds,'rate_bounds_km_s':rate_bounds,
        'angular_prior_sigma_in_tracklet_standard_deviations':angular_prior_sigma,
        'min_perigee_alt_km':min_perigee_alt_km},diagnostics)
