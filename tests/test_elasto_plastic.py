import unittest
import numpy as np
import matplotlib.pyplot as plt
from algorithms.ElastoPlastic import ElastoPlastic, generate_strain_path

class TestGenerateStrainPath(unittest.TestCase):
    def test_valid_inputs(self):
        """测试有效输入下的输出形状和范围"""
        # normal parameters
        strain = generate_strain_path(max_strain=0.06, n_steps=1000)
        self.assertEqual(len(strain), 1000)
        self.assertTrue(np.all(strain >= -0.06*1.1))
        self.assertTrue(np.all(strain <= 0.06*1.1))

        # test defalt parameter
        strain_default = generate_strain_path()
        self.assertEqual(len(strain_default), 25000)

    def test_invalid_inputs(self):
        """if parameters are invalid"""
        with self.assertRaises(ValueError):
            generate_strain_path(max_strain=-0.1)  # max_strain below zero
        with self.assertRaises(ValueError):
            generate_strain_path(n_steps=0)        # zero step

    def test_pattern_following(self):
        """test if the path is correct"""
        # compute the points manually
        points = np.array([0, 0.6, -0.8, 0.95, -0.7])
        scaled_points = 0.1 * points  # max_strain=0.1
        expected_values = np.interp(
            np.linspace(0, 4, 5),  # check 5 points
            np.arange(5),
            scaled_points
        )
        generated = generate_strain_path(max_strain=0.1, n_steps=5)
        np.testing.assert_array_almost_equal(generated, expected_values, decimal=5)

    def test_plot_generation(self):
        """test the plot function"""
        generate_strain_path(plot=True)

class TestElastoPlastic(unittest.TestCase):
    def setUp(self):
        """test the initialization"""
        self.E = 200000  # MPa
        self.Yi = 400    # MPa
        self.H = 10000   # MPa
        self.strain_path = np.linspace(0, 0.06, 1000)

    def test_invalid_initialization(self):
        """test invalid initicalization"""
        with self.assertRaises(ValueError):
            ElastoPlastic(self.E, self.H, self.Yi, mode='invalid')
        with self.assertRaises(ValueError):
            ElastoPlastic(self.E, -1000, self.Yi, mode='isotropic')

    def test_elastic_behavior(self):
        """test elastic stage"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='isotropic')
        small_strain = np.linspace(0, 0.001, 10)  # samll strain
        material.apply_loading(small_strain)
        
        # test whether the stress satifies the hookie's law
        expected_stress = self.E * small_strain
        np.testing.assert_allclose(material.stress_history, expected_stress, rtol=1e-5)
        self.assertEqual(material.epsilon_p, 0.0)  # no plasctic deformation

    def test_isotropic_hardening(self):
        """test isotropic hardening"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='isotropic')
        material.apply_loading(self.strain_path)
        
        # test if the yield stress increases
        final_yield_stress = material.Yi + material.H * material.epsilon_p
        self.assertAlmostEqual(material.Yn, final_yield_stress, places=2)
        
        # test if the last stress equals to the new yield stress
        self.assertAlmostEqual(material.stress_history[-1], final_yield_stress, places=2)

    def test_kinematic_hardening(self):
        """test kinematic hardening"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='kinematic')
        # First stretch to 0.06, then compress to -0.06.
        strain = np.concatenate([np.linspace(0, 0.06, 500), np.linspace(0.06, -0.06, 500)])
        material.apply_loading(strain)
        
        # inverse load
        compressive_stress = material.stress_history[-1]
        expected_yield = -material.Yi +  material.alpha_n
        self.assertAlmostEqual(compressive_stress, expected_yield, delta=10)

    def test_history_consistency(self):
        """the loading history should be the same"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='isotropic')
        material.apply_loading(self.strain_path)
        self.assertEqual(len(material.stress_history), len(material.strain_history))
        self.assertEqual(material.strain_history[0], 0.0)  # initisal strain should be 0

    def test_plot_function(self):
        """test the plot function is valid"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='isotropic')
        material.apply_loading(self.strain_path)
        material.plot_curve(save_path="test_elasto_plot.png")

if __name__ == '__main__':
    unittest.main()