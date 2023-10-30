import pandas as pd
import numpy as np


## check how train and test on this optimization makes sense
def get_eigenvalues_and_eigenvectors_denoising(returns, denoise_cov_matrix = False, plot_optimization = False):
    if denoise_cov_matrix:
        cov_matrix = get_denoised_cov_matrix_from_returns(returns, plot_optimization)
    else: 
        cov_matrix = returns.corr()
    
    D, S = np.linalg.eigh(cov_matrix)

    eigen_portfolios = []
    for i in range(len(S[0])):
        _ev = S[:,i]
        
        positive_sum = np.abs(np.sum([pos for pos in _ev if pos >= 0]))
        negative_sum = np.abs(np.sum([neg for neg in _ev if neg < 0]))
        
        normalized_weights = []
        for _ev_item_ in _ev:
            if _ev_item_ >= 0:
                normalized_weights.append(_ev_item_/positive_sum)
            else:
                normalized_weights.append(_ev_item_/negative_sum)
                
        eigen_portfolios.append({'eigen_value':D[i], 'weights': normalized_weights})
    
    return D, S, eigen_portfolios




## check how train and test on this optimization makes sense
def get_eigenvalues_and_eigenvectors(returns, is_cov = False):
    if is_cov:
        matrix = returns.cov()
    else: 
        matrix = returns.corr()
    

    D, S = np.linalg.eigh(matrix)

    eigen_portfolios = []
    for i in range(len(S[0])):
        _ev = S[:,i]
        
        positive_sum = np.abs(np.sum([pos for pos in _ev if pos >= 0]))
        negative_sum = np.abs(np.sum([neg for neg in _ev if neg < 0]))
        
        normalized_weights = []
        for _ev_item_ in _ev:
            if _ev_item_ >= 0:
                normalized_weights.append(_ev_item_/positive_sum)
            else:
                normalized_weights.append(_ev_item_/negative_sum)
                
        eigen_portfolios.append({'eigen_value':D[i], 'eigen_vector': _ev, 'normalized_weights': normalized_weights})
    
    return eigen_portfolios

#D, S = getPCA(cov_matrix) sorting this makes sense in this step? according to the tests, order doesn matter! but eigenvalues of the corr and coov matrix are different


## different normalization
        #otal_variance = np.sum(np.sqrt(np.power(S[:,i],2))) # this normalizes eigenvectors to the total variance explained, as removes negative sign
        #ep_ = S[:,i] / np.sum(S[:,i])
        #p_ = S[:,i] / total_variance