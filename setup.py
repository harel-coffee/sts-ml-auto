# Imports: third party
from setuptools import setup, find_packages

setup(
    name="ml4sts",
    version="0.0.1",
    description="Machine Learning for Society for Thoracic Surgery outcomes",
    url="https://github.com/aguirre-lab/cardiac-surgery",
    python_requires=">=3.8",
    install_requires=[],
    packages=find_packages(),
    entry_points={"console_scripts": ["ml4sts = ml4sts.recipes:main"]},
)
