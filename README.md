# UCSD-MLE-ZTF
Repo for UCSD MLE bootcamp capstone project.
The notebooks do the following
1. Collect ZTF variable stars light curve data either from ZTF DR2 database or from http://variables.cn:88/ website. 
1. Calculate statistical features like mean, median deviation, skewness, kurtosis etc.
1. Develop Classifiers for variable star types using these features. 
1. Compare Classifiers for different variable star types and different classifier types

#Setup
1. Download equal number of light curves for all variable stars by running notebook "Download_Processed_LCs.ipynb". The downloaded data is stored in an SQLite database
1. Extract features from the downloaded light curves by running "FeaturesFromXiaoDianLC.ipynb". The features are save in a .csv files
1. Train and compare classifiers by running notebook "OneVSallClassifier-comparison.ipynb"
