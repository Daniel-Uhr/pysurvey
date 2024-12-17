from setuptools import setup, find_packages

setup(
    name="pysurvey",
    version="0.1.0",
    description="A Python package for complex survey analysis",
    author="Daniel de Abreu Pereira Uhr",
    author_email="daniel.uhr@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "statsmodels>=0.12.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
