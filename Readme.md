# Sea-ice impacts inter-annual variability in bloom phenology and carbon export

This is a repository for the data analyses performed in Giddy et al. (submitted to GRL). Its purpose is to demonstrate the workflow and record all analyses performed in accompaniment with the manuscript.

This code is subject to change during the review process. 

### Summary 

The study aims to quantify and understand drivers of interannual variability in carbon export in the MIZ, using novel high resolution observations from sensors mounted on gliders. Variability in Primary Production and export is linked to variability in sea ice, but this link is not linear. We find that remineralization is an important variable in determining carbon export and is influenced by community composition and the density structure of the water column.  

### Workflow

The analyses (located in the folder notebooks) follows as:  

1) Characterisation of the region   (Figure 1) - [01_Studysite.ipynb](notebooks/01_Studysite.ipynb)
2) Interannual phenology based on satellite and reanalysis products as well as two SOCCOM bgc-argo floats   (Figure 2) - [02_Phenology.ipynb](notebooks/02_Phenology.ipynb)
3) Calculation of Primary Productivity from glider deployments  - [03_PrimaryProductivity.ipynb](notebooks/03_PrimaryProductivity.ipynb)
4) Calculation of large particle carbon export flux from the glider deployments. - [04_Export.ipynb](notebooks/04_Export.ipynb)
5) Comparison of Primary Prroductivity models to export - [05_PP_export_methods_comparison.ipynb](/notebooks/05_PP_export_methods_comparison.ipynb)
6) Analysis of the variation of PP and export in the two high resolution glider deployments   (Figure 3) - [05_VerticalFlux.ipynb](/notebooks/05_VerticalFlux.ipynb)

The conda environment which lists the dependencies for this code is provide.

Much of the analyses was performed with the aid of tools prewritten in [GliderTools](https://github.com/GliderToolsCommunity/GliderTools). 

The Primary Production CbPM model was adapted from the code written by [Lionel Arteaga](https://github.com/artlionel/SOCCOM_BGC_Float_data_public).  

While some of this code is written generically, most will require adaptation if is it to be used on different data. The functions which were developed specifically for this work are located in the /src folder and separated into PrimaryProductionTools.py (called in the notebook 03_PrimaryProductivity) and ExportTools.py (called in the notebook 04_Export.ipynb). Some of the figures in the manuscript were concatenated separately such that the output in the notebooks here do not reproduce the same figure formats as seen in the manuscript.     

### Data sources

The raw data will be made available via NCEI in the coming weeks. 

Below is a list of auxillary data sources that are used in the analysis:

   - [Biogeochemical ARGO](http://www.argo.ucsd.edu)
   - [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)
   - [Sea Ice Concentration (OSI-450 and OSI-430-b)](http://osisaf.met.no/p/ice/ice_conc_reprocessed.html)
   - [Sea Ice Extent](https://nsidc.org/data/G02135/versions/3)
   - [Thin Sea-ice Thickness](https://seaice.uni-bremen.de/data)
   - [Chlorophyll a](https://www.oceancolour.org/)
   - [Photosynthetically Active Radiation](https://oceandata.sci.gsfc.nasa.gov/)
    
