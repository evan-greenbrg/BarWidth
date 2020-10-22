# Quantifying bankfull flow width using preserved bar clinoforms from fluvial strata 
### A method to find the widths of river channels and the widths of channel-bar clinoforms.

This repository is in companion to the manuscript. It contains all sensitivity tests, workflows and base codes.

### Script files: 
ExtractRiverCenterlines.py: Given landsat Band 6, and Band 3 images and an EPSG, will produce a csv of all water-covered coordinates (the output of the RivaMap program).
getXsections.py: Produce data file of all river cross-sections at given interval. Can run this from command line. Uses yaml files (examples found in exampleInputs/sectionParams.yaml). Allows you to pick channel widths mannually or automatically.
getBarStats.py: Pull bar widths from the river-cross sections. Can fit sigmoids automatically or manually. Example input file found at /exampleInputs/barParams.yaml.
dataProcessing.py: Cleans the data produced from getBarStats.py and aggregates all rivers into single datafile
Analysis.py: Performs the Bayesian Regression and produces data figures.

### Example input files:
sectionParams.yaml:
    DEMpath: path to DEM file to be used in analysis
    CenterlinePath: path to txt or csv to be used
    esaPath: path to ESA surface water file
    CenterlineSmoothing: parameter to smooth centerline to make cross-sections easy to pull
    SectionLength: length of cross section
    SectionSmoothing: Smoothing parameter for the cross-sections
    OutputRoot: Where to export the cross-section data
    manual: True for manual widths, False for automatic widths
barParams.yaml:
    xPath: Path to cross-section object (output of getSections.py)
    coordPath: path to txt or csv of centerline
    bar_path: path to upstream-downstream pairs of bars
    demPAth: path to DEM file
    outputRoot: location to export bar-width data
    depth: target depth for river interpolation
    interpolate: True if you want to interpolate the bathymetry
    mannual: True if you want to manually fit the bar
    
