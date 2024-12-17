# pysurvey

**pysurvey** is a Python library for analyzing **complex survey designs**. Inspired by the `survey` package in R, it provides tools for weighted statistics, hypothesis testing, and generalized linear models for survey data.

---

## Installation

To install the package directly from GitHub, use:

```bash
pip install git+https://github.com/Daniel-Uhr/pysurvey.git
```

---

## Features
The `pysurvey` package implements the following functionalities:

1. Survey Design Management:
  * Create survey designs with weights, strata, and replicate weights.

2. Weighted Statistics:
* svymean: Compute weighted means.
* svytotal: Compute weighted totals.
* svyquantile: Compute weighted quantiles.
* svyvar: Compute weighted variances.

3. Hypothesis Testing:
* svychisq: Weighted chi-squared test for independence.
* svyttest: Weighted t-test for two groups.

4. Ratio and Proportions:
* svyratio: Compute weighted ratios between two variables.
* svyciprop: Compute weighted proportions with confidence intervals.

5. Subgroup Analysis:
* svyby: Apply weighted functions to subgroups.

6. Calibration and Replication:
* Weight calibration and bootstrap-based replicate weights.

---
## Example Usage

1. Define a Survey Design

```bash
import pandas as pd
from pysurvey.design import SurveyDesign

# Example dataset
data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'weight': [1.5, 2.0, 2.5, 3.0, 3.5],
    'strata': ['A', 'A', 'B', 'B', 'C'],
    'x': [10, 20, 30, 40, 50],
    'y': [5, 15, 25, 35, 45]
})

# Initialize Survey Design
design = SurveyDesign(data, ids='id', weights='weight', strata='strata')
```

2. Compute Weighted Statistics

```bash
from pysurvey.statistics import svymean, svytotal, svyvar

# Compute weighted mean and total
mean_result = svymean(data, data['weight'], ['x', 'y'])
total_result = svytotal(data, data['weight'], ['x', 'y'])

print("Weighted Means:", mean_result)
print("Weighted Totals:", total_result)

```
--- 

3. Perform Hypothesis Testing


```bash
from pysurvey.statistics import svychisq, svyttest

# Weighted Chi-Square Test
chisq_result = svychisq(data, data['weight'], group='strata', variable='x')
print("Chi-Square Test:", chisq_result)

# Weighted T-Test
ttest_result = svyttest(data, data['weight'], variable='x', group='strata')
print("T-Test Result:", ttest_result)

```

---

4. Compute Proportions and Ratios

```bash
from pysurvey.statistics import svyratio, svyciprop

# Compute Weighted Ratio
ratio_result = svyratio(data, data['weight'], numerator='x', denominator='y')
print("Weighted Ratio:", ratio_result)

# Weighted Proportion with Confidence Interval
data['binary'] = [0, 1, 1, 0, 1]
prop_result = svyciprop(data, data['weight'], 'binary')
print("Weighted Proportion:", prop_result)
```


--- 

4. Compute Proportions and Ratios

```bash
from pysurvey.statistics import svyratio, svyciprop

# Compute Weighted Ratio
ratio_result = svyratio(data, data['weight'], numerator='x', denominator='y')
print("Weighted Ratio:", ratio_result)

# Weighted Proportion with Confidence Interval
data['binary'] = [0, 1, 1, 0, 1]
prop_result = svyciprop(data, data['weight'], 'binary')
print("Weighted Proportion:", prop_result)

```

---

5. Subgroup Analysis

```bash
from pysurvey.statistics import svyby

# Apply weighted mean function to subgroups
by_result = svyby(data, data['weight'], 'strata', lambda d, w: svymean(d, w, 'x'))
print("Weighted Means by Strata:", by_result)
```


---
## Contributing

Contributions are welcome! If you'd like to add new features or improve the library, feel free to fork the repository and submit a pull request.

---
## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
## Author
Developed by Daniel de Abreu Pereira Uhr.
Contact: daniel.uhr@gmail.com

