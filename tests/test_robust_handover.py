import unittest
import numpy as np
from handover.robust_station_overlap import unit_vectors,elevations,load_states,pointing
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class HandoverTests(unittest.TestCase):
    def test_units(self):
        u=unit_vectors(np.array([0.,90.]),np.array([0.,0.]));self.assertTrue(np.allclose(np.linalg.norm(u,axis=1),1))
    def test_zenith(self):
        u=unit_vectors(np.array([0.]),np.array([0.]));self.assertAlmostEqual(float(elevations(8000*u,u)[0,0]),90,places=5)
    def test_modes(self):
        for mode,n in [('known',1),('single',44)]:
            epoch,states,w=load_states(ROOT/'examples/mc_fitted_5_min.npz',mode)
            self.assertEqual(len(states),n);self.assertAlmostEqual(w.sum(),1)
if __name__=='__main__':unittest.main()
