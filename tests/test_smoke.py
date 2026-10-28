import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'sky_search_prototype'))

class TestRepository(unittest.TestCase):
    def test_imports(self):
        import orbit_toy, orbit_fit_multipass, orbit_mc_fitted
        import orbit_stage_a, orbit_stage_b, orbit_global_dashboard
        import orbit_connected_regions, orbit_station_search, sky_search_maps

    def test_example_ensembles(self):
        from sky_search_maps import load_orbits
        for mode, file in [('single', 'mc_fitted_5_min.npz'),('global','stage_b_3_min.npz')]:
            states,weights,families,epoch=load_orbits(ROOT/'examples'/file,mode)
            self.assertEqual(states.shape[1],6)
            self.assertEqual(len(states),len(weights))
            self.assertAlmostEqual(float(weights.sum()),1.,places=12)
            self.assertTrue(np.isfinite(states).all())

    def test_field_of_view(self):
        from sky_search_maps import optimal_pointing
        unit=np.array([[1.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        center,score,mask=optimal_pointing(unit,np.array([True,True,True]),np.array([.2,.3,.5]),1.)
        self.assertAlmostEqual(score,.5,places=12)
        self.assertEqual(int(mask.sum()),2)

if __name__ == '__main__':unittest.main()

class TestRangeEpoch(unittest.TestCase):
    def test_range_at_independent_epoch(self):
        from optional_modes.known_range_search import condition_range
        from orbit_station_search import Station
        from orbit_toy import propagate, station_state
        source=ROOT/'examples'/'mc_fitted_5_min.npz'
        with np.load(source) as data:
            t=np.datetime64(str(data['epoch']),'ns')
            states=np.asarray(data['states'],float)
        station=Station('Bern',46.95,7.44,.55)
        at=t+np.timedelta64(600,'s')
        result,rho,ess=condition_range(source,20000.,1000.,station,measurement_epoch=str(at))
        r,_=propagate(states[:,:3].copy(),states[:,3:].copy(),600.)
        sr,_=station_state(at,46.95,7.44,.55)
        np.testing.assert_allclose(rho,np.linalg.norm(r-sr,axis=1),atol=1e-8)
        self.assertAlmostEqual(result['weights'].sum(),1.,places=12)
        self.assertTrue(ess>0)
        self.assertEqual(np.datetime64(result['range_measurement_epoch'],'ns'),at)
        _,rho0,_=condition_range(source,20000.,1000.,station)
        self.assertGreater(np.max(np.abs(rho-rho0)),1.)
