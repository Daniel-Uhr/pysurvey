import numpy as np


def svymean(data, weights, variables):
    """
    Compute weighted means for specified variables.

    Parameters:
    - data: pd.DataFrame
        The input data.
    - weights: pd.Series
        Column of weights.
    - variables: list
        List of variable names to compute means for.

    Returns:
    - dict: Weighted means for the specified variables.
    """
    if not isinstance(variables, list):
        variables = [variables]
    means = {}
    for var in variables:
        if var not in data.columns:
            raise ValueError(f"Variable '{var}' not found in the data.")
        means[var] = np.average(data[var], weights=weights)
    return means


def svytotal(data, weights, variables):
    """
    Compute weighted totals for specified variables.

    Parameters:
    - data: pd.DataFrame
        The input data.
    - weights: pd.Series
        Column of weights.
    - variables: list
        List of variable names to compute totals for.

    Returns:
    - dict: Weighted totals for the specified variables.
    """
    if not isinstance(variables, list):
        variables = [variables]
    totals = {}
    for var in variables:
        if var not in data.columns:
            raise ValueError(f"Variable '{var}' not found in the data.")
        totals[var] = np.sum(data[var] * weights)
    return totals


def svyquantile(data, weights, variables, quantiles):
    """
    Compute weighted quantiles for specified variables.

    Parameters:
    - data: pd.DataFrame
        The input data.
    - weights: pd.Series
        Column of weights.
    - variables: list
        List of variable names to compute quantiles for.
    - quantiles: list
        List of quantiles (between 0 and 1).

    Returns:
    - dict: Weighted quantiles for the specified variables.
    """
    if not isinstance(variables, list):
        variables = [variables]
    if not isinstance(quantiles, list):
        raise ValueError("Quantiles must be provided as a list.")
    results = {}
    for var in variables:
        if var not in data.columns:
            raise ValueError(f"Variable '{var}' not found in the data.")
        sorted_indices = np.argsort(data[var])
        sorted_values = data[var].iloc[sorted_indices].values
        sorted_weights = weights.iloc[sorted_indices].values
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        quantile_results = {}
        for q in quantiles:
            target = q * total_weight
            idx = np.searchsorted(cum_weights, target)
            quantile_results[q] = sorted_values[idx]
        results[var] = quantile_results
    return results


def svyvar(data, weights, variables):
    """
    Compute weighted variance for specified variables.

    Parameters:
    - data: pd.DataFrame
        The input data.
    - weights: pd.Series
        Column of weights.
    - variables: list
        List of variable names to compute variance for.

    Returns:
    - dict: Weighted variances for the specified variables.
    """
    if not isinstance(variables, list):
        variables = [variables]
    variances = {}
    for var in variables:
        if var not in data.columns:
            raise ValueError(f"Variable '{var}' not found in the data.")
        mean = np.average(data[var], weights=weights)
        variance = np.average((data[var] - mean) ** 2, weights=weights)
        variances[var] = variance
    return variances

