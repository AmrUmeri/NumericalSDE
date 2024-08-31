from setuptools import setup, find_packages

setup(
    name='numerical_sde_lib',
    version='1.1',
    description='A simple package for solving SDEs numerically',
    author='Amr Umeri',
    author_email='amr.umeri@outlook.com',
    url='https://github.com/AmrUmeri/NumericalSDE',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    python_requires='>=3.1',
)