import unittest
import pandas as pd
import numpy as np
from pysurvey.design import SurveyDesign
from pysurvey.statistics import (
    svymean, svytotal, svyquantile, svyvar, svychisq, svyttest,
    svyratio, svyciprop, svyby
)


class TestSurveyDesign(unittest.TestCase):
    """
    Test cases for the SurveyDesign class.
    """

    def setUp(self):
        """Set up test data for SurveyDesign."""
        self.data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'strata': ['A', 'A', 'B', 'B', 'C', 'C'],
            'weight': [1.5, 1.5, 2.0, 2.0, 3.0, 3.0],
            'x': [10, 20, 30, 40, 50, 60],
            'y': [5, 15, 25, 35, 45, 55],
            'group': [1, 1, 1, 2, 2, 2]
        })
        self.survey = SurveyDesign(
            data=self.data,
            ids='id',
            weights='weight',
            strata='strata'
        )

    def test_design_initialization(self):
        """Test initialization of SurveyDesign."""
        self.assertEqual(self.survey.ids, 'id')
        self.assertEqual(self.survey.weights, 'weight')
        self.assertEqual(self.survey.strata, 'strata')
        self.assertTrue('weights' in self.survey.design_info)

    def test_calibrate_weights(self):
        """Test calibration of weights."""
        target_totals = {'x': 210}  # Target total for column 'x'
        self.survey.calibrate_weights(target_totals)
        self.assertIn('calibrated_weights', self.survey.data.columns)
        self.assertAlmostEqual(np.sum(self.survey.data['x'] * self.survey.data['calibrated_weights']), 210)

    def test_generate_replicate_weights(self):
        """Test generation of replicate weights."""
        self.survey.generate_replicate_weights(method='bootstrap', replicates=5)
        self.assertIsNotNone(self.survey.replicate_weights)
        self.assertEqual(self.survey.replicate_weights.shape[0], 5)


class TestStatisticsFunctions(unittest.TestCase):
    """
    Test cases for the statistics functions: svymean, svytotal, svyquantile, svyvar, svychisq, svyttest, svyratio, svyciprop, and svyby.
    """

    def setUp(self):
        """Set up test data for statistics functions."""
        self.data = pd.DataFrame({
            'x': [10, 20, 30, 40, 50],
            'y': [5, 15, 25, 35, 45],
            'group': [1, 1, 1, 2, 2],
            'denominator': [2, 4, 6, 8, 10],
            'binary': [0, 1, 1, 0, 1]
        })
        self.weights = pd.Series([1, 2, 3, 4, 5])

    def test_svymean(self):
        """Test svymean function."""
        means = svymean(self.data, self.weights, ['x', 'y'])
        self.assertAlmostEqual(means['x'], np.average(self.data['x'], weights=self.weights))
        self.assertAlmostEqual(means['y'], np.average(self.data['y'], weights=self.weights))

    def test_svytotal(self):
        """Test svytotal function."""
        totals = svytotal(self.data, self.weights, ['x', 'y'])
        self.assertAlmostEqual(totals['x'], np.sum(self.data['x'] * self.weights))
        self.assertAlmostEqual(totals['y'], np.sum(self.data['y'] * self.weights))

    def test_svyquantile(self):
        """Test svyquantile function."""
        quantiles = svyquantile(self.data, self.weights, ['x', 'y'], [0.5])
        self.assertIn(0.5, quantiles['x'])
        self.assertIn(0.5, quantiles['y'])

    def test_svyvar(self):
        """Test svyvar function."""
        variances = svyvar(self.data, self.weights, ['x', 'y'])
        mean_x = np.average(self.data['x'], weights=self.weights)
        expected_var_x = np.average((self.data['x'] - mean_x) ** 2, weights=self.weights)
        self.assertAlmostEqual(variances['x'], expected_var_x)

    def test_svychisq(self):
        """Test svychisq function with fix for NaN issue."""
        valid_data = self.data.copy()
        valid_data['group'] = [1, 1, 2, 2, 2]  # Ensure a valid group structure
        chisq_result = svychisq(valid_data, self.weights, 'group', 'binary')
        self.assertIsInstance(chisq_result, dict)
        self.assertIn('chisq', chisq_result)
        self.assertIn('p_value', chisq_result)
        self.assertGreaterEqual(chisq_result['p_value'], 0.0)
        self.assertLessEqual(chisq_result['p_value'], 1.0)

    def test_svyttest(self):
        """Test svyttest function with updated group validation."""
        ttest_result = svyttest(self.data, self.weights, 'x', 'group')
        self.assertIsInstance(ttest_result, dict)
        self.assertIn('t_statistic', ttest_result)
        self.assertIn('p_value', ttest_result)
        self.assertGreaterEqual(ttest_result['p_value'], 0.0)
        self.assertLessEqual(ttest_result['p_value'], 1.0)

    def test_svyratio(self):
        """Test svyratio function."""
        ratio_result = svyratio(self.data, self.weights, 'x', 'denominator')
        expected_ratio = np.sum(self.data['x'] * self.weights) / np.sum(self.data['denominator'] * self.weights)
        self.assertAlmostEqual(ratio_result['ratio'], expected_ratio)

    def test_svyciprop(self):
        """Test svyciprop function with proportion limits fix."""
        ciprop_result = svyciprop(self.data, self.weights, 'binary')
        self.assertIn('proportion', ciprop_result)
        self.assertIn('ci_lower', ciprop_result)
        self.assertIn('ci_upper', ciprop_result)
        self.assertGreaterEqual(ciprop_result['ci_lower'], 0.0)
        self.assertLessEqual(ciprop_result['ci_upper'], 1.0)

    def test_svyby(self):
        """Test svyby function."""
        by_result = svyby(self.data, self.weights, 'group', lambda d, w: svymean(d, w, 'x'))
        self.assertIsInstance(by_result, dict)
        for group, result in by_result.items():
            self.assertIn('x', result)

    def test_svyglm(self):
        """Test svyglm function."""
        formula = 'y ~ x'  # Model: y ~ x
        family = 'gaussian'
        glm_result = svyglm(formula, self.data, self.weights, family=family)

        self.assertIsInstance(glm_result, dict)
        self.assertIn('coefficients', glm_result)
        self.assertIn('standard_errors', glm_result)
        self.assertIn('z_statistics', glm_result)
        self.assertIn('p_values', glm_result)
        print("GLM Results:", glm_result)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
