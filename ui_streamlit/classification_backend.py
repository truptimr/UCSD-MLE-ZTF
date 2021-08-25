# classification backend
# Load libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle # allows to save differnt trained models of the same classifier object
import time
import streamlit as st


@st.cache
def query_lightcurve_XD(SourceID): 
    """
    Download data for a single source from Xiao Dian's website. Source is identified using SourceID
    """
    url = 'http://variables.cn:88/seldataz.php?SourceID=' + str(SourceID)   
    try:
        lc_complete = pd.read_csv(url, header='infer')
        lc = lc_complete.drop(columns = ['SourceID','flag'])
    except:
        lc_complete = pd.DataFrame()
        lc = pd.DataFrame()
    return lc, lc_complete


@st.cache
def query_lightcurve_DR(RA, Dec): 
    """
    Download data for a single source from DR2 dataset. Source is identified using RA and Dec location
    """
    circle_radius = 0.0028 # 1 arcsec = 0.00028 degress
    t_format = "ipac_table"
    table_format = "FORMAT=" + str(t_format)
    flag_mask = 32768
    mask = "BAD_CATFLAGS_MASK=" + str(flag_mask)
    collect="COLLECTION="+"ztf_dr2"
    numobs = "NOBS_MIN=20"
#     filter_band = "g"
    label = []
    SourceID =[]
    start_time = time.time()
    ra = RA
    dec = Dec
    circle = "POS=CIRCLE"+"+"+str(ra)+"+"+str(dec)+"+"+str(circle_radius)
#     band = "BANDNAME="+ filter_band
    params = circle + "&" +  mask + "&" + numobs + "&" + collect + "&" + table_format

    try:
        url= "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?" + params
        lc_complete = pd.read_csv(url, header=None, delim_whitespace=True, skiprows=55) # extract data
        header = pd.read_csv(url, header=None, sep='|', skiprows=50,usecols=range(1,25), nrows=1)
        lc_complete.columns = header.iloc[0].str.strip()
        lc = lc_complete[['ra','dec','hjd','mag','magerr','filtercode']]
        lc.columns=['RAdeg', 'DEdeg', 'HJD', 'mag', 'e_mag', 'band']
        lc.replace({'zg':'g'},inplace = True)
        lc.replace({'zr':'r'},inplace = True)
        val = lc.loc[:,'HJD']-2.4e6
        lc.loc[:,'HJD'] = val
    except:
        lc_complete = pd.DataFrame()
        lc = pd.DataFrame()

    return lc, lc_complete


def plot_lc(lc):
    """
    Function to plot the light curves
    """
    data1 = lc[lc['band']=='r']
    data2 = lc[lc['band']=='g']

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
    ax = axs[0]
    ax.errorbar(data2['HJD'],data2['mag'],yerr = data2['e_mag'],fmt='g.')
    ax.invert_yaxis() # smaller magnitude means brighter stars, so invert the axis
    ax.set_xlabel('time in HJD')
    ax.set_ylabel('magnitude')
    ax.set_title('Green Filter (g band)')

    ax = axs[1]
    ax.errorbar(data1['HJD'],data1['mag'],yerr = data1['e_mag'],fmt='r.')
    ax.invert_yaxis() # smaller magnitude means brighter stars, so invert the axis
    ax.set_xlabel ('time in HJD')
    ax.set_ylabel('magnitude')
    ax.set_title('Red Filter (r filter)')
    

    fig.tight_layout(pad=3.0)
    fig.suptitle('Measured Light Curve', fontsize=16)
    st.pyplot(fig)


# def weighted_mean(mag,mag_err):
#     mag2 = (mag_err*mag_err) # mag err square
#     mag2_inv = 1/mag2.values; # take inverse of the values
#     w = pd.Series(mag2_inv) # covert it back to s series
#     sw = w.sum() # sum of weights
#     wmag = mag*w # multiply magnitude with weights
#     wmean = wmag.sum()/sw # weighted mean
#     return wmean

def weighted_mean(mag,e_mag):
    w = 1/(e_mag*e_mag)
    sw = np.sum(w)
    wmag = w*mag
    wmean = np.sum(wmag)/sw
    return wmean


# welsh J, K statistics
def welsh_staton(mag_series,wmean):
    N = len(mag_series)
    d_i = N/(N-1)*(mag_series - wmean) # replace mean by weighted mean
    d_i1 = d_i.shift(periods=-1)
    d_i1.fillna(0, inplace = True)
    Pi = d_i*d_i1
    Pi_val = Pi.values
    Psign = np.sign(Pi_val)
    Jval = Psign*np.sqrt(np.abs(Pi_val))
    J = np.sum(Jval) 
    K1 = abs(d_i.values)/N
    K2 = np.sqrt(1/N*np.sum(d_i.values*d_i.values))
    K = np.sum(K1*K2)
    return J, K


def calculate_features(lc):
    """
    Calculate features for a light curve passed as a dataframe.
    """
    g_mean = []
    g_wmean = [] # weighted mean
    g_MAD = []
    g_IQR = []
    g_f60 = []
    g_f70 = []
    g_f80 = []
    g_f90 = []
    g_skew = []
    g_kurtosis = []
    g_welsh_K = []
    g_welsh_J = []

    r_mean = []
    r_wmean = [] # weighted mean
    r_MAD = []
    r_IQR = []
    r_f60 = []
    r_f70 = []
    r_f80 = []
    r_f90 = []
    r_skew = []
    r_kurtosis = []
    r_welsh_K = []
    r_welsh_J = []
    
    if len(lc) >1:
        
        dfg = lc.loc[lc["band"] == "g"]
        dfr = lc.loc[lc["band"] == "r"]
        if len(dfg) > 1:
            N = len(dfg)
            wmean_temp = weighted_mean(dfg.mag.values,dfg.e_mag.values)
            K_temp, J_temp =  welsh_staton(dfg.mag, wmean_temp )
            g_mean.append(dfg.mag.mean())
            g_wmean.append(wmean_temp) 
            deviation = abs(dfg.mag - dfg.mag.median())
            g_MAD.append(deviation.median())
            g_IQR.append(dfg.mag.quantile(0.75) - dfg.mag.quantile(0.25))
            g_f60.append(dfg.mag.quantile(0.80) - dfg.mag.quantile(0.2))
            g_f70.append(dfg.mag.quantile(0.85) - dfg.mag.quantile(0.15))
            g_f80.append(dfg.mag.quantile(0.9) - dfg.mag.quantile(0.10))
            g_f90.append(dfg.mag.quantile(0.95) - dfg.mag.quantile(0.05))
            g_skew.append(dfg.mag.skew())
            g_kurtosis.append(dfg.mag.kurtosis())
            g_welsh_J.append(J_temp)
            g_welsh_K.append(K_temp)
        else:
            g_mean.append(np.NaN)
            g_wmean.append(np.NaN) 
            g_MAD.append(np.NaN)
            g_IQR.append(np.NaN)
            g_f60.append(np.NaN)
            g_f70.append(np.NaN)
            g_f80.append(np.NaN)
            g_f90.append(np.NaN)
            g_skew.append(np.NaN)
            g_kurtosis.append(np.NaN)
            g_welsh_J.append(np.NaN)
            g_welsh_K.append(np.NaN)
                
        if len(dfr) >1:
            N = len(dfr)
            wmean_temp = weighted_mean(dfr.mag.values,dfr.e_mag.values)
            K_temp, J_temp =  welsh_staton(dfr.mag, wmean_temp )
            r_mean.append(dfr.mag.mean())
            r_wmean.append(wmean_temp) 
            deviation = abs(dfr.mag - dfr.mag.median())
            r_MAD.append(deviation.median())
            r_IQR.append(dfr.mag.quantile(0.75) - dfr.mag.quantile(0.25))
            r_f60.append(dfr.mag.quantile(0.80) - dfr.mag.quantile(0.2))
            r_f70.append(dfr.mag.quantile(0.85) - dfr.mag.quantile(0.15))
            r_f80.append(dfr.mag.quantile(0.9) - dfr.mag.quantile(0.10))
            r_f90.append(dfr.mag.quantile(0.95) - dfr.mag.quantile(0.05))
            r_skew.append(dfr.mag.skew())
            r_kurtosis.append(dfr.mag.kurtosis())
            r_welsh_J.append(J_temp)
            r_welsh_K.append(K_temp)
        else:
            r_mean.append(np.NaN)
            r_wmean.append(np.NaN) 
            r_MAD.append(np.NaN)
            r_IQR.append(np.NaN)
            r_f60.append(np.NaN)
            r_f70.append(np.NaN)
            r_f80.append(np.NaN)
            r_f90.append(np.NaN)
            r_skew.append(np.NaN)
            r_kurtosis.append(np.NaN)
            r_welsh_J.append(np.NaN)
            r_welsh_K.append(np.NaN)

    else:
        g_mean.append(np.NaN)
        g_wmean.append(np.NaN) 
        g_MAD.append(np.NaN)
        g_IQR.append(np.NaN)
        g_f60.append(np.NaN)
        g_f70.append(np.NaN)
        g_f80.append(np.NaN)
        g_f90.append(np.NaN)
        g_skew.append(np.NaN)
        g_kurtosis.append(np.NaN)
        g_welsh_J.append(np.NaN)
        g_welsh_K.append(np.NaN)
        r_mean.append(np.NaN)
        r_wmean.append(np.NaN) 
        r_MAD.append(np.NaN)
        r_IQR.append(np.NaN) 
        r_f60.append(np.NaN)
        r_f70.append(np.NaN)
        r_f80.append(np.NaN)
        r_f90.append(np.NaN)
        r_skew.append(np.NaN)
        r_kurtosis.append(np.NaN)
        r_welsh_J.append(np.NaN)
        r_welsh_K.append(np.NaN)
        
    # del features
    features = pd.DataFrame()
    N = 1

    # g filter data
    features['g_mean'] = g_mean[0:N]
    features['g_wmean'] = g_wmean[0:N]
    features['g_MAD'] = g_MAD[0:N]
    features['g_IQR'] = g_IQR[0:N]
    features['g_f60'] = g_f60[0:N]
    features['g_f70'] = g_f70[0:N]
    features['g_f80'] = g_f80[0:N]
    features['g_f90'] = g_f90[0:N]
    features['g_skew'] = g_skew[0:N]
    features['g_kurtosis'] = g_kurtosis[0:N]
    features['g_welsh_J'] = g_welsh_J[0:N]
    features['g_welsh_K'] = g_welsh_K[0:N]

    # r filter data
    features['r_mean'] = r_mean[0:N]
    features['r_wmean'] = r_wmean[0:N]
    features['r_MAD'] = r_MAD[0:N]
    features['r_IQR'] = r_IQR[0:N]
    features['r_f60'] = r_f60[0:N]
    features['r_f70'] = r_f70[0:N]
    features['r_f80'] = r_f80[0:N]
    features['r_f90'] = r_f90[0:N]
    features['r_skew'] = r_skew[0:N]
    features['r_kurtosis'] = r_kurtosis[0:N]
    features['r_welsh_J'] = r_welsh_J[0:N]
    features['r_welsh_K'] = r_welsh_K[0:N]

    return features



def prediction_probabilty(features):
    """
    Predict probability for each of the 9 variable types using pre calculated features.
    """
    prob={}
    label = ['BYDra', 'EW', 'SR', 'RSCVN', 'RR', 'DSCT', 'EA', 'Mira', 'RRc']
    prob_pd = pd.DataFrame(columns=['Probability'],index=label)
    if np.isnan(features.iloc[0,:].values).all():
        pass
    else:
        for variable_type in label:
            print(variable_type)
            name = 'XGBoost'
            filename = '../pickles/'+ name+'_'+variable_type+'.pkl'
            clf = pickle.load(open(filename, 'rb'))
            predict_proba = clf.predict_proba(features)
            prob[variable_type] = round(predict_proba[0,0],2)
    #         prob[variable_type] = clf.predict_proba(features)
        prob_pd['Probability']=prob.values()
    return prob_pd


@st.cache
def true_label(ID):
    """
    Find true star type in the labeled data set
    """
    ## open label data table
    widths = (8,7,4,13,43)
    header_pd = pd.read_fwf('../databases/Labeled_data.txt', widths = widths,skiprows=7, nrows=27)
    labeled_data = pd.read_csv('../databases/Labeled_data.txt', header=None, delim_whitespace=True, skiprows=36) # extract data
    labeled_data.columns = header_pd.iloc[:,3]
    true_label = labeled_data.loc[labeled_data['SourceID']==ID,'Type']
    return true_label.values[0]