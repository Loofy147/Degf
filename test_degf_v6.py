import unittest
import numpy as np
from degf_v6 import run_trt_benchmark, run_hallucination_f1, run_sgs2_timing, run_thermo_shift, get_k_laws

class TestDEGFv6(unittest.TestCase):
    def test_trt(self):
        res = run_trt_benchmark()
        self.assertEqual(res['score'], 0.806)
        self.assertEqual(res['pass_count'], 9)
        # 0.49 gap is sufficient for simulation
        self.assertGreater(res['gap'], 0.45)

    def test_hallu(self):
        res = run_hallucination_f1()
        self.assertEqual(res['f1'], 1.0)
        self.assertEqual(res['fp'], 0)

    def test_sgs2(self):
        res = run_sgs2_timing()
        self.assertGreater(res['math'], res['inductive'])
        self.assertGreater(res['deductive'], res['inductive'])

    def test_thermo(self):
        res = run_thermo_shift()
        self.assertAlmostEqual(res['delta'], 0.111)
        self.assertGreater(res['g_end'], res['g_start'])

    def test_klaws(self):
        kd, kr = get_k_laws(12)
        self.assertAlmostEqual(kd, 0.8129, places=3)
        self.assertAlmostEqual(kr, 1.2371, places=3)

if __name__ == '__main__':
    unittest.main()
