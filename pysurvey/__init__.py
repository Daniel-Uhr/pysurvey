# pysurvey/__init__.py

from .design import SurveyDesign
from .statistics import (
    svymean, svytotal, svyquantile, svychisq, svyttest,
    svyratio, svyciprop, svyby
)

__version__ = "0.1.0"
__author__ = "Daniel de Abreu Pereira Uhr"
