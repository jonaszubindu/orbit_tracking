"""Toy angles-only orbit search from a single optical tracklet.

Dependencies: numpy, matplotlib (optional plotting), and tracklet.py.
Units: km, km/s, radians. Approximate spherical Earth + GMST rotation;
RA/Dec are assumed geocentric-inertial-axis equatorial directions (J2000-like).
No light-time, precession/nutation, refraction, drag or J2.
"""
from dataclasses import dataclass
import numpy as np
from tracklet import fit_tracklet

MU = 398600.4418  # km^3/s^2
RE = 6378.137     # km
OMEGA = 7.2921150e-5  # rad/s
ARCSEC = 180 * 3600 / np.pi


def unit_and_rate(fit):
    a, d = np.deg2rad([fit.ra_deg, fit.dec_deg])
    ad, dd = np.deg2rad([fit.ra_rate_deg_s, fit.dec_rate_deg_s])
    u = np.array([np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)])
    ua = np.array([-np.cos(d)*np.sin(a), np.cos(d)*np.cos(a), 0.])
    ud = np.array([-np.sin(d)*np.cos(a), -np.sin(d)*np.sin(a), np.cos(d)])
    return u, ua*ad + ud*dd


def julian_date(t):
    # UTC used as UT1 in this deliberately approximate model.
    return 2440587.5 + (np.datetime64(t, 'ns') - np.datetime64('1970-01-01T00:00:00', 'ns')) / np.timedelta64(1, 's') / 86400.


def station_state(t, lat_deg, lon_deg, altitude_km=0.):
    lat, lon = np.deg2rad([lat_deg, lon_deg])
    jd = julian_date(t)
    T = (jd - 2451545.) / 36525.
    gmst_deg = 280.46061837 + 360.98564736629*(jd-2451545.) + 0.000387933*T*T - T**3/38710000.
    theta = np.deg2rad(gmst_deg % 360.) + lon
    radius = RE + altitude_km
    r = radius*np.array([np.cos(lat)*np.cos(theta), np.cos(lat)*np.sin(theta), np.sin(lat)])
    v = np.cross(np.array([0., 0., OMEGA]), r)
    return r, v


def initial_state(fit, station, ranges, range_rates):
    u, udot = unit_and_rate(fit)
    rr, vv = station
    r = rr[None, :] + ranges[:, None]*u[None, :]
    v = vv[None, :] + range_rates[:, None]*u[None, :] + ranges[:, None]*udot[None, :]
    return r, v


def orbital_filter(r, v, min_perigee_alt_km=100.):
    rad = np.linalg.norm(r, axis=1)
    energy = .5*np.sum(v*v, axis=1) - MU/rad
    h2 = np.sum(np.cross(r, v)**2, axis=1)
    eccentricity = np.sqrt(np.maximum(0., 1 + 2*energy*h2/MU**2))
    semimajor = np.where(energy < 0, -MU/(2*np.minimum(energy, -1e-30)), np.nan)
    perigee = np.where(energy < 0, semimajor*(1-eccentricity), np.nan)
    return (energy < 0) & (perigee > RE+min_perigee_alt_km), perigee-RE


def propagate(r, v, dt, max_step=5.):
    """Vectorized two-body RK4 for all candidate states."""
    steps = max(1, int(np.ceil(abs(dt)/max_step)))
    h = float(dt)/steps
    def accel(x):
        n = np.linalg.norm(x, axis=1)
        return -MU*x / n[:, None]**3
    for _ in range(steps):
        k1r, k1v = v, accel(r)
        k2r, k2v = v+h*k1v/2, accel(r+h*k1r/2)
        k3r, k3v = v+h*k2v/2, accel(r+h*k2r/2)
        k4r, k4v = v+h*k3v, accel(r+h*k3r)
        r = r+h*(k1r+2*k2r+2*k3r+k4r)/6
        v = v+h*(k1v+2*k2v+2*k3v+k4v)/6
    return r, v


@dataclass
class SearchResult:
    ranges_km: np.ndarray
    range_rates_km_s: np.ndarray
    chi2: np.ndarray
    weights: np.ndarray
    admissible: np.ndarray
    best_state_r_km: np.ndarray
    best_state_v_km_s: np.ndarray
    best_range_km: float
    best_range_rate_km_s: float
    best_chi2: float
    epoch: np.datetime64


def search_orbits(times, ra_deg, dec_deg, lat_deg, lon_deg, altitude_km=0.,
                  sigma_arcsec=5., range_bounds_km=(200., 60000.),
                  range_rate_bounds_km_s=(-10., 10.), n_range=80, n_rate=81,
                  min_perigee_alt_km=100.):
    """Grid search of rho and rho-dot with full-arc angular residual scoring.

    Uniform grid prior in range and range rate; posterior weights are conditional
    on this prior and on the bound-orbit/perigee filters. Not a calibrated orbit
    covariance. RA uncertainty is assumed in RA coordinate angle, not RA*cos(Dec).
    """
    fit = fit_tracklet(times, ra_deg, dec_deg, sigma_arcsec, sigma_arcsec)
    t = np.asarray(times, dtype='datetime64[ns]')
    rho_axis = np.geomspace(*range_bounds_km, n_range)
    rate_axis = np.linspace(*range_rate_bounds_km_s, n_rate)
    rho, rhodot = np.meshgrid(rho_axis, rate_axis, indexing='ij')
    rho, rhodot = rho.ravel(), rhodot.ravel()
    r0, v0 = initial_state(fit, station_state(fit.epoch, lat_deg, lon_deg, altitude_km), rho, rhodot)
    admissible, _ = orbital_filter(r0, v0, min_perigee_alt_km)
    if not np.any(admissible):
        raise ValueError('No admissible bound orbit on grid. Expand the grid or relax constraints.')
    chi2 = np.full(len(rho), np.inf)
    idx = np.flatnonzero(admissible)
    score = np.zeros(len(idx))
    observed_ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    observed_dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    for ti, a, d in zip(t, observed_ra, observed_dec):
        dt = float((ti-fit.epoch)/np.timedelta64(1, 's'))
        r, _ = propagate(r0[idx].copy(), v0[idx].copy(), dt)
        station_r, _ = station_state(ti, lat_deg, lon_deg, altitude_km)
        los = r-station_r
        ra_model = np.arctan2(los[:, 1], los[:, 0])
        dec_model = np.arctan2(los[:, 2], np.linalg.norm(los[:, :2], axis=1))
        dra = np.arctan2(np.sin(ra_model-a), np.cos(ra_model-a))
        score += ((dra*ARCSEC/sigma_arcsec)**2 + ((dec_model-d)*ARCSEC/sigma_arcsec)**2)
    chi2[idx] = score
    best = int(np.argmin(chi2))
    # A uniform prior per GRID POINT; logarithmic rho grid means nonuniform density in rho.
    w = np.zeros_like(chi2)
    w[idx] = np.exp(-.5*(score-np.min(score)))
    w /= w.sum()
    return SearchResult(rho, rhodot, chi2, w, admissible, r0[best], v0[best],
                        float(rho[best]), float(rhodot[best]), float(chi2[best]), fit.epoch)


def plot_search(result):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = result.admissible
    sc = ax.scatter(result.ranges_km[valid], result.range_rates_km_s[valid],
                    c=np.log10(result.weights[valid]+1e-300), s=10, cmap='viridis')
    ax.scatter([result.best_range_km], [result.best_range_rate_km_s], marker='x', s=90, c='red')
    ax.set_xscale('log')
    ax.set_xlabel('Topocentric range [km]')
    ax.set_ylabel('Range rate [km/s]')
    ax.set_title('Toy angles-only orbit grid: log10 posterior grid weights')
    fig.colorbar(sc, ax=ax, label='log10 grid weight')
    fig.tight_layout()
    return fig, ax


if __name__ == '__main__':
    # Generate internally consistent synthetic data from a circular-ish LEO state.
    lat, lon = 46.95, 7.44
    times = np.array(['2026-10-22T20:00:00', '2026-10-22T20:00:10',
                      '2026-10-22T20:00:20', '2026-10-22T20:00:30',
                      '2026-10-22T20:00:40', '2026-10-22T20:00:50'], dtype='datetime64[ns]')
    epoch = times[0]
    station_r, _ = station_state(epoch, lat, lon)
    up = station_r / np.linalg.norm(station_r)
    east = np.cross([0., 0., 1.], up)
    east /= np.linalg.norm(east)
    r_truth = station_r + 1000.*up
    v_truth = 7.35*east
    ra, dec = [], []
    for ti in times:
        dt = float((ti-epoch)/np.timedelta64(1, 's'))
        r, _ = propagate(r_truth[None, :], v_truth[None, :], dt)
        station_r_i, _ = station_state(ti, lat, lon)
        los = (r[0]-station_r_i)
        ra.append(np.rad2deg(np.arctan2(los[1], los[0])) % 360.)
        dec.append(np.rad2deg(np.arctan2(los[2], np.linalg.norm(los[:2]))))
    result = search_orbits(times, ra, dec, lat, lon)
    print('Best range [km]:', result.best_range_km)
    print('Best range rate [km/s]:', result.best_range_rate_km_s)
    print('Best chi-square:', result.best_chi2)
    print('Admissible states:', int(result.admissible.sum()))
    plot_search(result)
    import matplotlib.pyplot as plt
    plt.show()
