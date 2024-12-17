import numpy as np
import pandas as pd
from scipy.stats import chi2, t
from statsmodels.api import WLS, GLM, families


def svymean(data, weights, variables):
    """
    Compute weighted means for specified variables.
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


def svychisq(data, weights, group, variable):
    """
    Perform weighted chi-squared test for independence.
    """
    contingency_table = pd.crosstab(data[group], data[variable], values=weights, aggfunc='sum', normalize=False)
    row_totals = contingency_table.sum(axis=1)
    col_totals = contingency_table.sum(axis=0)
    total = contingency_table.sum().sum()

    expected = np.outer(row_totals, col_totals) / total
    observed = contingency_table.values
    chisq = np.sum((observed - expected) ** 2 / expected)
    dof = (len(row_totals) - 1) * (len(col_totals) - 1)
    p_value = chi2.sf(chisq, dof)

    return {'chisq': chisq, 'p_value': p_value}


def svyttest(data, weights, variable, group):
    """
    Perform weighted t-test for two groups.
    """
    unique_groups = data[group].unique()
    if len(unique_groups) != 2:
        raise ValueError("Group variable must have exactly two unique values.")

    group1 = data[data[group] == unique_groups[0]]
    group2 = data[data[group] == unique_groups[1]]

    weights1 = weights.loc[group1.index]
    weights2 = weights.loc[group2.index]

    mean1 = np.average(group1[variable], weights=weights1)
    mean2 = np.average(group2[variable], weights=weights2)

    var1 = np.average((group1[variable] - mean1) ** 2, weights=weights1)
    var2 = np.average((group2[variable] - mean2) ** 2, weights=weights2)

    n1, n2 = len(group1), len(group2)
    se = np.sqrt(var1 / n1 + var2 / n2)
    t_statistic = (mean1 - mean2) / se
    dof = n1 + n2 - 2
    p_value = 2 * t.sf(np.abs(t_statistic), dof)

    return {'t_statistic': t_statistic, 'p_value': p_value}


def svyratio(data, weights, numerator, denominator):
    """
    Compute weighted ratio of two variables.
    """
    if numerator not in data.columns or denominator not in data.columns:
        raise ValueError("Both numerator and denominator must be in the data.")
    num = np.sum(data[numerator] * weights)
    denom = np.sum(data[denominator] * weights)
    ratio = num / denom
    return {'ratio': ratio}


def svyciprop(data, weights, variable, alpha=0.05):
    """
    Compute confidence intervals for weighted proportions.
    """
    prop = np.average(data[variable], weights=weights)
    n = len(data)
    se = np.sqrt(prop * (1 - prop) / n)
    z = t.ppf(1 - alpha / 2, df=n - 1)
    lower = prop - z * se
    upper = prop + z * se
    return {'proportion': prop, 'ci_lower': lower, 'ci_upper': upper}


def svyby(data, weights, group, func):
    """
    Apply a weighted function to subgroups.
    """
    grouped = data.groupby(group)
    results = {}
    for name, group_data in grouped:
        group_weights = weights.loc[group_data.index]
        results[name] = func(group_data, group_weights)
    return results
