"""Fit a short RA/Dec optical tracklet and estimate its attributable covariance.

Angles are ICRS-like equatorial RA/Dec in degrees, timestamps are UTC ISO strings.
Measurement errors are independent between epochs and RA/Dec in this first version.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Attributable:
    epoch: np.datetime64
    ra_deg: float
    dec_deg: float
    ra_rate_deg_s: float
    dec_rate_deg_s: float
    covariance: np.ndarray  # [rad, rad, rad/s, rad/s] order
    residuals_arcsec: np.ndarray  # N x 2, [RA*cos(dec), Dec]
    chi2: float
    dof: int


def fit_tracklet(times, ra_deg, dec_deg, sigma_ra_arcsec=5.0, sigma_dec_arcsec=None):
    """Weighted linear fit; sigmas refer to RA coordinate angle and Dec angle.

    Pass scalar sigmas or one per observation. RA sigma is in coordinate angle,
    not projected on the tangent plane. Returns covariance in radians.
    """
    if sigma_dec_arcsec is None:
        sigma_dec_arcsec = sigma_ra_arcsec
    t = np.asarray(times, dtype='datetime64[ns]')
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    n = len(t)
    if n < 3 or ra.shape != (n,) or dec.shape != (n,):
        raise ValueError('Need at least 3 timestamps and matching RA/Dec arrays')
    if np.any(~np.isfinite(ra)) or np.any(~np.isfinite(dec)) or np.any(np.abs(dec) > 90):
        raise ValueError('Invalid RA/Dec values')
    seconds = (t - t[0]) / np.timedelta64(1, 's')
    if np.any(np.diff(seconds) <= 0):
        raise ValueError('Timestamps must be strictly increasing')
    dt = seconds - np.mean(seconds)
    epoch = t[0] + np.timedelta64(int(round(np.mean(seconds)*1e9)), 'ns')
    ra_unwrapped = np.unwrap(np.deg2rad(ra))
    dec_rad = np.deg2rad(dec)
    sig_ra = np.broadcast_to(np.asarray(sigma_ra_arcsec, dtype=float), (n,)) * np.pi / (180*3600)
    sig_dec = np.broadcast_to(np.asarray(sigma_dec_arcsec, dtype=float), (n,)) * np.pi / (180*3600)
    if np.any(sig_ra <= 0) or np.any(sig_dec <= 0):
        raise ValueError('All uncertainties must be positive')
    X = np.column_stack((np.ones(n), dt))
    def solve(y, sigma):
        A = X / sigma[:, None]
        b = y / sigma
        beta, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank != 2:
            raise ValueError('Degenerate timestamps')
        cov = np.linalg.inv(A.T @ A)
        resid = y - X @ beta
        return beta, cov, resid
    bra, cra, rra = solve(ra_unwrapped, sig_ra)
    bdec, cdec, rdec = solve(dec_rad, sig_dec)
    # Reorder [RA0, RA_rate, Dec0, Dec_rate] -> [RA0, Dec0, RA_rate, Dec_rate]
    C = np.zeros((4, 4))
    C[np.ix_([0, 2], [0, 2])] = cra
    C[np.ix_([1, 3], [1, 3])] = cdec
    chi2 = float(np.sum((rra / sig_ra)**2 + (rdec / sig_dec)**2))
    residuals = np.column_stack((rra*np.cos(bdec[0]), rdec)) * (180*3600/np.pi)
    return Attributable(epoch, float(np.rad2deg(bra[0]) % 360),
                        float(np.rad2deg(bdec[0])), float(np.rad2deg(bra[1])),
                        float(np.rad2deg(bdec[1])), C, residuals, chi2, 2*n-4)


if __name__ == '__main__':
    times = [f'2026-10-22T20:00:{s:02d}' for s in range(0, 60, 10)]
    ra = [359.98, 359.99, 0.00, 0.01, 0.02, 0.03]
    dec = [12.00, 12.005, 12.010, 12.015, 12.020, 12.025]
    fit = fit_tracklet(times, ra, dec, sigma_ra_arcsec=5.0)
    print('Epoch:', fit.epoch)
    print(f'RA, Dec: {fit.ra_deg:.6f}, {fit.dec_deg:.6f} deg')
    print(f'RA rate, Dec rate: {fit.ra_rate_deg_s:.8f}, {fit.dec_rate_deg_s:.8f} deg/s')
    print('1-sigma [arcsec, arcsec, arcsec/s, arcsec/s]:',
          np.sqrt(np.diag(fit.covariance))*180*3600/np.pi)
    print(f'Chi2 / dof: {fit.chi2:.2f} / {fit.dof}')
