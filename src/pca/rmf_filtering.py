import pandas as pd
import numpy as np

# fitting empirical distributions with gaussian kernels
from sklearn.neighbors import KernelDensity

# Optimization
from scipy.optimize import minimize
import matplotlib.pyplot as plt


"""
Code reference: Elements in Quantitative Finance - Machine Learning for Asset Managers

Methods for denoising covariance matrix
"""

## Fitting distr methods ###

"""
Code reference: Elements in Quantitative Finance - Machine Learning for Asset Managers

Methods for denoising covariance matrix
"""

## Fitting distr methods ###

def mpPDF(var, q, pts):
    # Marcenko-Pastur theoretical pdf
    # q=T/N
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5) ** 2
    
    eVal = np.linspace(eMin, eMax, pts)
    eVal = np.reshape(eVal, len(eVal))
    
    pdf = q / (2 * np.pi * var * eVal)*((eMax - eVal) * (eVal - eMin)) ** .5
    pdf = np.reshape(pdf,len(pdf))
    
    pdf = pd.Series(pdf, index=eVal)
    return pdf

def fitKDE(obs, bWidth = 0.25, kernel='gaussian', x = None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    
    if len(obs.shape) == 1:
        obs = obs.reshape(-1,1) 
    kde = KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    
    if x is None:
        x = np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:
        x=x.reshape(-1,1) 
        
    logProb = kde.score_samples(x) # log(density) 
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf


## Fitting empirical Marchenko-pastur distr thorugh optimization process - minimizing error betwen the theortical and the empirical
def errPDFs(var, eVal, q, bWidth, plot, pts=500):
    
    pdf0 = mpPDF(var, q, pts) # theoretical pdf 
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values) # empirical pdf 
    
    if plot:
        fig, axs = plt.subplots(figsize=(10, 6), constrained_layout=True)

        axs.plot(pdf0.index, pdf0, 'black','-')
        axs.plot(pdf1.index, pdf1, 'red','-')
        axs.hist(eVal, 50)
        axs.set_title('Portfolio Performance')

    # Fit error
    sse = np.sum((pdf1-pdf0)**2)
    return sse


def findMaxEval(eVal, q, bWidth, plot):
    # Find max random eVal by fitting Marcenko’s dist 
    
    # this method fails when few eigenvalues are considered
    
    # bandwitdh for cov matrix: .02
    out = minimize(lambda *x: errPDFs(*x), np.std(eVal), args= (eVal, q, bWidth, plot), bounds=((1E-5,.50-1E-5),))
    
    if out['success']:
        var=out['x'][0] 
    else:
        var=1 
    
    eMax = var*(1+(1./q)**.5)**2 
    
    return eMax,var


### Eigenvalues and eigenvectors of the cov matrix ###
def getPCA(matrix):
    # Get eVal,eVec from a Hermitian matrix 
    eVal,eVec = np.linalg.eigh(matrix) 
    indices = eVal.argsort()[::-1] # arguments for sorting eVal desc 
    eVal, eVec = eVal[indices],eVec[:,indices] 
    eVal = np.diagflat(eVal)
    return eVal,eVec


### Denoising taking place ####
def denoisedCorr(eVal, eVec, nFacts):
    # Remove noise from corr by fixing random eigenvalues 
    eVal_ = np.diag(eVal).copy() 
    #eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0]-nFacts)
    eVal_[nFacts:] = 0
    eVal_ = np.diag(eVal_)
    cov = np.dot(eVec,eVal_).dot(eVec.T) 
    #corr1=cov2corr(cov)
    return cov

def denoising_orchestrator(matrix, q, bandwidth, plot_optimization):
    # get eigenvalues and eigenvectors without filtering
    eVal, eVec = getPCA(matrix)
    
    # finding nFacts, which determines the cutoff treshold in the empricial fitted Marchenko-pastur distr.
    eMax, var = findMaxEval(np.diag(eVal), q, bandwidth, plot_optimization) 
    nFacts = eVal.shape[0] - np.diag(eVal)[::-1].searchsorted(eMax)
    
    # denoising covariance matrix
    denoised_matrix = denoisedCorr(eVal, eVec, nFacts)
    
    return denoised_matrix

def emax_orchestrator(matrix, q, bandwidth, plot_optimization):
    # get eigenvalues and eigenvectors without filtering
    eVal, eVec = getPCA(matrix)
    # finding nFacts, which determines the cutoff treshold in the empricial fitted Marchenko-pastur distr.
    eMax, var = findMaxEval(np.diag(eVal), q, bandwidth, plot_optimization) 
    return eMax


###### main ####
def get_denoised_corr_matrix_from_returns(returns, bandwidth = 0.15, plot = False):
    matrix = returns.corr() 
    q = float(returns.shape[0]) / float(returns.shape[1])
    
    return denoising_orchestrator(matrix, q,  bandwidth = bandwidth,  plot_optimization = plot)

def get_denoised_cov_matrix_from_returns(returns,  plot = False):
    matrix = returns.cov() 
    q = float(returns.shape[0]) / float(returns.shape[1])
    
    return denoising_orchestrator(matrix, q, bandwidth = bandwidth, plot_optimization = plot)


def get_eigen_values_cutting_point(returns, bandwidth = 0.15, plot = False, is_corr = True):
    if is_corr:  
        matrix = returns.corr()
    else:
        matrix = returns.cov()
    
    q = float(returns.shape[0]) / float(returns.shape[1])
    return emax_orchestrator(matrix, q, bandwidth = bandwidth, plot_optimization = plot)
    