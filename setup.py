from setuptools import setup

setup(
    name='BarWidth',
    version='0.1.0',
    author='Evan Greenberg',
    author_email='egreenberg@ucsb.edu',
    packages=['BarWidth'],
    license='LICENSE.txt',
    description='Toolset to accompany Manuscript titled: Quantifying bankfull flow width using preserved bar clinoforms from fluvial strata',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "pandas",
        "rasterio",
        "scipy",
        "matplotlib",
        "sklearn",
        "arviz",
        "pymc3",
        "rivamap",
        "geopandas",
        "shapely",
        "gdal",
        "fiona",
        "pycrs",
    ],
)
