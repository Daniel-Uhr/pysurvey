# pysurvey/__init__.py

# Importação das classes e funções principais do pacote

from .design import SurveyDesign
from .statistics import svymean, svytotal, svyquantile, svychisq, svyttest, svyglm
from .utils import validate_input, stratify_data, replicate_weights

# Metadados do pacote
__version__ = "0.1.0"
__author__ = "Daniel de Abreu Pereira Uhr"
__email__ = "daniel.uhr@gmail.com"
__license__ = "MIT"

# Tornar as principais funcionalidades acessíveis
__all__ = [
    "SurveyDesign",
    "svymean",
    "svytotal",
    "svyquantile",
    "svychisq",
    "svyttest",
    "svyglm",
    "validate_input",
    "stratify_data",
    "replicate_weights",
]
