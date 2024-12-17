import pandas as pd
import numpy as np
from scipy.optimize import minimize


class SurveyDesign:
    """
    Class to define complex survey designs with clustering, stratification, and weighting.
    Supports finite population corrections (fpc) and replicate weights for variance estimation.
    """

    def __init__(self, data, ids=None, weights=None, strata=None, fpc=None):
        """
        Initialize a survey design.

        Parameters:
        - data: pd.DataFrame
            The input data.
        - ids: str or list of str, optional
            Cluster identifiers (Primary Sampling Units).
        - weights: str, optional
            Column name for sampling weights.
        - strata: str, optional
            Column name for stratification.
        - fpc: str, optional
            Column name for finite population correction.
        """
        self.data = self._validate_data(data)
        self.ids = ids
        self.weights = weights
        self.strata = strata
        self.fpc = fpc
        self.replicate_weights = None  # Placeholder for replicate weights
        self.design_info = {}

        # Process design parameters
        self._process_design()

    def _validate_data(self, data):
        """Validate input data as a pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        return data

    def _process_design(self):
        """Validate and process design parameters."""
        if self.weights:
            if self.weights not in self.data.columns:
                raise ValueError(f"Column '{self.weights}' for weights not found in the data.")
            self.design_info['weights'] = self.data[self.weights]
        if self.strata:
            if self.strata not in self.data.columns:
                raise ValueError(f"Column '{self.strata}' for strata not found in the data.")
            self.design_info['strata'] = self.data[self.strata]
        if self.fpc:
            if self.fpc not in self.data.columns:
                raise ValueError(f"Column '{self.fpc}' for finite population correction not found.")
            self.design_info['fpc'] = self.data[self.fpc]
        if self.ids:
            if isinstance(self.ids, str) and self.ids not in self.data.columns:
                raise ValueError(f"Column '{self.ids}' for cluster IDs not found in the data.")
            elif isinstance(self.ids, list):
                for col in self.ids:
                    if col not in self.data.columns:
                        raise ValueError(f"Column '{col}' for cluster IDs not found in the data.")
            self.design_info['ids'] = self.ids

    def calibrate_weights(self, target_totals):
        """
        Calibrate weights to match target totals for specific variables.

        Parameters:
        - target_totals: dict
            Dictionary where keys are column names and values are target totals.
        """
        def objective(weights):
            return np.sum(weights ** 2)

        constraints = []
        for var, target in target_totals.items():
            if var not in self.data.columns:
                raise ValueError(f"Variable '{var}' not found in the data.")
            constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w * self.data[var]) - target})

        # Initial weights
        initial_weights = self.design_info['weights'] if 'weights' in self.design_info else np.ones(len(self.data))

        # Solve optimization
        result = minimize(objective, initial_weights, constraints=constraints)
        if not result.success:
            raise ValueError("Calibration failed to converge.")
        self.data['calibrated_weights'] = result.x
        print("Weights successfully calibrated.")

    def generate_replicate_weights(self, method='bootstrap', replicates=50):
        """
        Generate replicate weights for variance estimation.

        Parameters:
        - method: str, optional
            Method to generate replicate weights ('bootstrap' or 'brr').
        - replicates: int, optional
            Number of replicate weights to generate.
        """
        if method not in ['bootstrap', 'brr']:
            raise ValueError("Method must be 'bootstrap' or 'brr'.")

        replicate_weights = []
        n = len(self.data)

        if method == 'bootstrap':
            for _ in range(replicates):
                indices = np.random.choice(range(n), size=n, replace=True)
                replicate_weights.append(self.data[self.weights].iloc[indices].values)

        elif method == 'brr':
            half_size = n // 2
            for _ in range(replicates):
                indices = np.concatenate((np.random.choice(range(half_size), half_size, replace=False),
                                          np.random.choice(range(half_size, n), half_size, replace=False)))
                replicate_weights.append(self.data[self.weights].iloc[indices].values)

        self.replicate_weights = np.array(replicate_weights)
        print(f"{replicates} replicate weights generated using {method} method.")

    def summary(self):
        """Print summary of the survey design."""
        print("Survey Design Summary:")
        for key, value in self.design_info.items():
            print(f"{key.capitalize()}: {value.name if isinstance(value, pd.Series) else value}")
        if 'calibrated_weights' in self.data.columns:
            print("Calibrated weights present.")
        if self.replicate_weights is not None:
            print(f"Replicate weights generated: {self.replicate_weights.shape[0]} replicates.")
