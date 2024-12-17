import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, t
import statsmodels.api as sm

def svyglm(formula, data, weights, family='gaussian'):
    """
    Fit a Generalized Linear Model (GLM) with survey weights.

    Parameters:
        formula (str): A patsy-style formula defining the model.
        data (DataFrame): The dataset containing the variables.
        weights (array-like): Survey weights.
        family (str): Distribution family ('gaussian', 'binomial', 'poisson', 'gamma').

    Returns:
        dict: Summary including coefficients, standard errors, z-statistics, and p-values.
    """
    # Map family strings to statsmodels families
    families_map = {
        'gaussian': sm.families.Gaussian(),
        'binomial': sm.families.Binomial(),
        'poisson': sm.families.Poisson(),
        'gamma': sm.families.Gamma()
    }

    if family not in families_map:
        raise ValueError(f"Unsupported family: {family}. Supported families: {list(families_map.keys())}")

    # Fit the GLM model
    model = sm.GLM.from_formula(formula, data=data, freq_weights=weights, family=families_map[family])
    results = model.fit()

    # Prepare model summary
    summary = {
        'coefficients': results.params.to_dict(),
        'standard_errors': results.bse.to_dict(),
        'z_statistics': results.tvalues.to_dict(),
        'p_values': results.pvalues.to_dict(),
        'deviance': results.deviance,
        'null_deviance': results.null_deviance,
        'aic': results.aic
    }
    return summary


def svymean(data, weights, variables):
    """Compute weighted means for specified variables."""
    variables = [variables] if not isinstance(variables, list) else variables
    return {var: np.average(data[var], weights=weights) for var in variables}


def svytotal(data, weights, variables):
    """Compute weighted totals for specified variables."""
    variables = [variables] if not isinstance(variables, list) else variables
    return {var: np.sum(data[var] * weights) for var in variables}


def svyquantile(data, weights, variables, quantiles):
    """Compute weighted quantiles for specified variables."""
    if not isinstance(quantiles, list):
        raise ValueError("Quantiles must be a list.")

    results = {}
    for var in ([variables] if isinstance(variables, str) else variables):
        sorted_idx = np.argsort(data[var])
        sorted_values = data[var].iloc[sorted_idx].values
        sorted_weights = weights.iloc[sorted_idx].values
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        
        results[var] = {
            q: sorted_values[np.searchsorted(cum_weights, q * total_weight)]
            for q in quantiles
        }
    return results


def svyvar(data, weights, variables):
    """Compute weighted variance for specified variables."""
    results = {}
    for var in ([variables] if isinstance(variables, str) else variables):
        mean = np.average(data[var], weights=weights)
        variance = np.average((data[var] - mean) ** 2, weights=weights)
        results[var] = variance
    return results


def svychisq(data, weights, group, variable):
    """Perform weighted chi-squared test for independence."""
    contingency = pd.crosstab(data[group], data[variable], values=weights, aggfunc='sum', normalize=False)
    row_totals = contingency.sum(axis=1)
    col_totals = contingency.sum(axis=0)
    expected = np.outer(row_totals, col_totals) / row_totals.sum()
    observed = contingency.values
    chisq = np.sum((observed - expected) ** 2 / expected)
    dof = (len(row_totals) - 1) * (len(col_totals) - 1)
    p_value = chi2.sf(chisq, dof)
    return {'chisq': chisq, 'p_value': p_value}


def svyttest(data, weights, variable, group):
    """Perform weighted t-test for two groups."""
    groups = data[group].unique()
    if len(groups) != 2:
        raise ValueError("Group variable must have exactly two unique values.")
    
    g1, g2 = [data[data[group] == grp] for grp in groups]
    w1, w2 = weights.loc[g1.index], weights.loc[g2.index]
    mean1, mean2 = np.average(g1[variable], weights=w1), np.average(g2[variable], weights=w2)
    var1, var2 = np.average((g1[variable] - mean1) ** 2, weights=w1), np.average((g2[variable] - mean2) ** 2, weights=w2)
    n1, n2 = len(w1), len(w2)
    se = np.sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean1 - mean2) / se
    dof = n1 + n2 - 2
    p_value = 2 * t.sf(np.abs(t_stat), dof)
    return {'t_statistic': t_stat, 'p_value': p_value}


def svyratio(data, weights, numerator, denominator):
    """Compute weighted ratio of two variables."""
    num = np.sum(data[numerator] * weights)
    denom = np.sum(data[denominator] * weights)
    return {'ratio': num / denom}


def svyciprop(data, weights, variable, confidence=0.95):
    """Compute weighted proportion and confidence interval."""
    prop = np.average(data[variable], weights=weights)
    n_eff = np.sum(weights)**2 / np.sum(weights**2)
    se = np.sqrt(prop * (1 - prop) / n_eff)
    z = norm.ppf(1 - (1 - confidence) / 2)
    return {'proportion': prop, 'ci_lower': max(0, prop - z * se), 'ci_upper': min(1, prop + z * se)}


def svyby(data, weights, group, func):
    """Apply a weighted function to subgroups."""
    return {name: func(subgroup, weights.loc[subgroup.index]) for name, subgroup in data.groupby(group)}

__all__ = [
    "svymean", "svytotal", "svyquantile", "svyvar", "svychisq", "svyttest",
    "svyratio", "svyciprop", "svyby", "svyglm"
]
