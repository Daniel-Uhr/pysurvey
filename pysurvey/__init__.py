# pysurvey/__init__.py

from .design import SurveyDesign
from .statistics import (
    svymean, svytotal, svyquantile, svyvar, svychisq, svyttest,
    svyratio, svyciprop, svyby, svyglm
)

__version__ = "0.1.0"
__author__ = "Daniel de Abreu Pereira Uhr"
