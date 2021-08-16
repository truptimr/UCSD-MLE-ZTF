import pandas as pd
import streamlit as st
import classification_backend as cb
import sqlalchemy as db


web_text = """
This is an interactive webpage which predicts the variable type of a given light curve from Zwicky Transient Facility (ZTF) database. 
The prediction is done using XG Boost classifiers.
The classifiers consists of 1 vs All XG Boost classifiers for 9 different types of periodic variable stars.
The classifiers were trained using labeled ZTF DR2 data set obtained from http://variables.cn:88/
"""


feature_explanation = """
			Feature list  
			- **mean** : mean of the light curve.
			- **wmean** : weight mean of the light curve.
			- **MAD**: deviation about the median.
			- **IQR** : inter quartile percentile of the light curve.
			- **f60** : 60 percentile  of light curve.
			- **f70** : 70 percentile of the light curve.
			- **f80** : 80 percentile of the light curve.
			- **f90** : 90 percentile of the light curve.
			- **skew** : skewness of the light curve.
			- **kurtosis** : kurtosis of the light curve.
			- **welsk_k, welsh_j** : welsh and staton J and K statistics of the light curve.
			- **g_**: g band filter.
			- **r_** : r band filter.
			"""

variable_type_explanation = """
			## Glossary of different type of variable stars

			The 9 different types of variable stars in the data are as follows:  

			1. **BYDra**: BY Draconis variables are variable stars of late spectral types, usually K or M, and typically belong to the main sequence. 
			1. **EW** : EW-type eclipsing binaries (EWs) are W Ursae Majoris-type eclipsing variables with periods shorter than one day.
			1. **SR** : semi-regular variables
			1. **RSCVN** : An RS Canum Venaticorum variable is a type of variable star. The variable type consists of close binary stars having active chromospheres which can cause large stellar spots.
			1. **RR** :RR Lyrae variables are periodic variable stars, commonly found in globular clusters. They are used as standard candles to measure (extra) galactic distances, assisting with the cosmic distance ladder. 
			1. **DSCT**: A Delta Scuti variable (sometimes termed dwarf cepheid when the V-band amplitude is larger than 0.3 mag.) is a subclass of young pulsating star. These variables as well as classical cepheids 
			1. **EA** : Algol (Beta Persei)-type eclipsing systems. Binaries with spherical or slightly ellipsoidal components. 
			1. **Mira** : Mira variables are a class of pulsating stars characterized by very red colours, pulsation periods longer than 100 days, and amplitudes greater than one magnitude in infrared and 2.5 magnitude at visual wavelengths. 
			1. **RRc** : RR Lyrae variable stars of subclass c. 
			"""

References = """
Zwicky Transient Facility  

1. [ZTF Website](https://www.ztf.caltech.edu/)
1. [Jan van Roestel et al 2021 AJ 161 267] (https://iopscience.iop.org/article/10.3847/1538-3881/abe853/meta?casa_token=8xxK5DlzXYsAAAAA:kELEfby98rcABefbQtaO20Iiwf-0SojgTt1tRcuqhk--xDVv_r1O5XVv2wZHQpLOV-njnptl5A)
1. [Xiaodian Chen et al 2020 ApJS **249** 18] (https://iopscience.iop.org/article/10.3847/1538-4365/ab9cae/meta)  

Code  

1. [Pandas](https://pandas.pydata.org/)  
1. [sklearn](https://scikit-learn.org/stable/)  
1. [streamlit](https://streamlit.io/)  
1. [Git repo](https://github.com/truptimr/UCSD-MLE-ZTF)  
"""

Help = """
1. Select a dataset to query.  
1. Input light source paramaters. 
1. Query the data.
1. (Optional) Save light curve in a SQ Lite database locally. The table index of the light curve in the database is of the format <RA>_<Dec> where RA and dec are floats with 2 decimal places.
1. Make a predition.
1. Tip: Uncheck all boxes before switching between datasets
"""

## open local SQL lite database connections
engine = db.create_engine('sqlite:///LightSource.db', echo=False)
sqlite_connection = engine.connect()






def prediction(lc, NA):
	"""
	Do prediction and display results
	"""
	features = cb.calculate_features(lc)
	"""
	## Features
	"""
	cols = features.columns.values
	df1 = features.loc[:,cols[0:8]]
	df2 = features.loc[:,cols[8:16]]
	df3 = features.loc[:,cols[16:24]]
	st.write(df1)
	st.write(df2)
	st.write(df3)
	if st.checkbox('Feature Explanation'):
		st.markdown(feature_explanation)
	prob = cb.prediction_probabilty(features)
	"""
	## Prediction Probability
	"""
	probT = prob.T
	st.write(probT.style.format("{:.2}"))
	max_prob = prob.Probability.max()
	if max_prob>0.5 and not(NA):
		star = prob.idxmax(axis=0, skipna=True).values[0]
		pred_str = '**The most likely star type for the lightcurve is ' + star + '**'
		st.markdown(pred_str)
	if max_prob<=0.5 or NA:
		st.markdown('**The lightcurve does not belog to any of the 9 periodic variables types**')
	if st.checkbox('Variable Type Explanation'):
		st.markdown(variable_type_explanation)

### The Webpage ####

st.title("ZTF data variable star classifier")

st.markdown(web_text)

if st.checkbox('References'):
	st.markdown(References)

if st.checkbox('Help'):
	st.markdown(Help)

st.sidebar.header('Light Source Datasets')
st.sidebar.text("""Uncheck all boxes before 
switching between datasets""")

if st.sidebar.checkbox('ZTF DR2 Dataset'):
	st.sidebar.text('''Input source location RA and 
DEC with 10 arcsec accuracy
''' )
	RA = st.sidebar.number_input('Right Acsecion (Degree)', min_value=0.0, max_value=360.0, value=72.0, step=0.01)
	Dec = st.sidebar.number_input('Declination Acsecion (Degree)', min_value=0.0, max_value=90.0, value = 23.00, step=0.01)
	if st.sidebar.checkbox('Query Location'):
		lc1, lc_complete1 = cb.query_lightcurve_DR(RA, Dec)
		if lc1.empty:
			st.sidebar.text("No data available")
			NA = True
		else:
			st.sidebar.text("Data is available")
			cb.plot_lc(lc1)
			NA = False
			if st.sidebar.checkbox('Save to Database  (Optional)'):
				index = str(round(lc1.RAdeg[0],3))+'_'+ str(round(lc1.DEdeg[0],3))
				lc1.to_sql(index, sqlite_connection, if_exists = 'replace')


	if st.sidebar.checkbox('Predict') and ('lc1' in locals()):
		prediction(lc1,NA)
		


if st.sidebar.checkbox('Xiaodian Chen Dataset'):
	st.sidebar.text('''Use Source ID to query dataset
from http://variables.cn:88/
SourceID range = 1 to 781602
''' )
	ID = st.sidebar.number_input('SourceID (integer)', min_value=1, max_value=781602, value=1, step=1)
	if st.sidebar.checkbox('Query ID'):
		lc2, lc_complete2 = cb.query_lightcurve_XD(ID)
		if lc2.empty:
			st.sidebar.text("No data available")
			NA = True
		else:
			st.sidebar.text("Data is available")
			cb.plot_lc(lc2)
			NA = False
			if st.sidebar.checkbox('Save to Database (Optional)'):
				index = str(round(lc2.RAdeg[0],3))+'_'+ str(round(lc2.DEdeg[0],3))
				lc2.to_sql(index, sqlite_connection, if_exists = 'replace')
	if st.sidebar.checkbox('Predict') and ('lc2' in locals()):
		prediction(lc2, NA)


		










