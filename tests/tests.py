import unittest
import pandas as pd
import numpy as np
from design import SurveyDesign
from statistics import svymean, svytotal, svyquantile, svyvar


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
            'y': [5, 15, 25, 35, 45, 55]
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
    Test cases for the statistics functions: svymean, svytotal, svyquantile, and svyvar.
    """
    
    def setUp(self):
        """Set up test data for statistics functions."""
        self.data = pd.DataFrame({
            'x': [10, 20, 30, 40, 50],
            'y': [5, 15, 25, 35, 45]
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


if __name__ == '__main__':
    unittest.main()
