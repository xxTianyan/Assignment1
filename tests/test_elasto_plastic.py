import unittest
import numpy as np
import matplotlib.pyplot as plt
from algorithms.ElastoPlastic import ElastoPlastic, generate_strain_path

class TestGenerateStrainPath(unittest.TestCase):
    def test_valid_inputs(self):
        """测试有效输入下的输出形状和范围"""
        # 正常参数
        strain = generate_strain_path(max_strain=0.06, n_steps=1000)
        self.assertEqual(len(strain), 1000)
        self.assertTrue(np.all(strain >= -0.06*1.1))
        self.assertTrue(np.all(strain <= 0.06*1.1))

        # 测试默认参数
        strain_default = generate_strain_path()
        self.assertEqual(len(strain_default), 25000)

    def test_invalid_inputs(self):
        """测试无效输入引发的错误"""
        with self.assertRaises(ValueError):
            generate_strain_path(max_strain=-0.1)  # 负max_strain
        with self.assertRaises(ValueError):
            generate_strain_path(n_steps=0)        # 零步长

    def test_pattern_following(self):
        """测试生成的路径是否遵循预设模式"""
        # 手动计算关键点
        points = np.array([0, 0.6, -0.8, 0.95, -0.7])
        scaled_points = 0.1 * points  # max_strain=0.1
        expected_values = np.interp(
            np.linspace(0, 4, 5),  # 精确匹配5个关键点
            np.arange(5),
            scaled_points
        )
        generated = generate_strain_path(max_strain=0.1, n_steps=5)
        np.testing.assert_array_almost_equal(generated, expected_values, decimal=5)

    def test_plot_generation(self):
        """测试绘图功能是否正常执行"""
        # 检查是否生成图像文件且不报错
        generate_strain_path(plot=True)

class TestElastoPlastic(unittest.TestCase):
    def setUp(self):
        """初始化测试用的材料参数"""
        self.E = 200000  # MPa
        self.Yi = 400    # MPa
        self.H = 10000   # MPa
        self.strain_path = np.linspace(0, 0.06, 1000)

    def test_invalid_initialization(self):
        """测试无效的初始化参数"""
        with self.assertRaises(ValueError):
            ElastoPlastic(self.E, self.H, self.Yi, mode='invalid')
        with self.assertRaises(ValueError):
            ElastoPlastic(self.E, -1000, self.Yi, mode='isotropic')

    def test_elastic_behavior(self):
        """测试弹性阶段的行为"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='isotropic')
        small_strain = np.linspace(0, 0.001, 10)  # 小应变，不触发屈服
        material.apply_loading(small_strain)
        
        # 验证应力是否符合胡克定律
        expected_stress = self.E * small_strain
        np.testing.assert_allclose(material.stress_history, expected_stress, rtol=1e-5)
        self.assertEqual(material.epsilon_p, 0.0)  # 无塑性应变

    def test_isotropic_hardening(self):
        """测试各向同性硬化的塑性行为"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='isotropic')
        material.apply_loading(self.strain_path)
        
        # 验证屈服应力是否增加
        final_yield_stress = material.Yi + material.H * material.epsilon_p
        self.assertAlmostEqual(material.Yn, final_yield_stress, places=2)
        
        # 验证最后一个应力是否等于更新后的屈服应力
        self.assertAlmostEqual(material.stress_history[-1], final_yield_stress, places=2)

    def test_kinematic_hardening(self):
        """测试随动硬化的包辛格效应"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='kinematic')
        # 先拉伸到0.06，再压缩到-0.06
        strain = np.concatenate([np.linspace(0, 0.06, 500), np.linspace(0.06, -0.06, 500)])
        material.apply_loading(strain)
        
        # 验证反向加载时的屈服应力降低（包辛格效应）
        compressive_stress = material.stress_history[-1]
        expected_yield = -material.Yi +  material.alpha_n
        self.assertAlmostEqual(compressive_stress, expected_yield, delta=10)  # 允许一定误差

    def test_history_consistency(self):
        """测试应力和应变历史的一致性"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='isotropic')
        material.apply_loading(self.strain_path)
        self.assertEqual(len(material.stress_history), len(material.strain_history))
        self.assertEqual(material.strain_history[0], 0.0)  # 初始应变应为0

    def test_plot_function(self):
        """测试绘图函数是否正常执行"""
        material = ElastoPlastic(self.E, self.H, self.Yi, mode='isotropic')
        material.apply_loading(self.strain_path)
        material.plot_curve(save_path="test_elasto_plot.png")

if __name__ == '__main__':
    unittest.main()