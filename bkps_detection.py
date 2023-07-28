import numpy as np
import ruptures as rpt
from joblib import delayed, Parallel

# --- Change point detection --- 


def get_bkps_pred(signal, beta = 0.4, gamma = None): 
    if gamma is None: 
        gamma = 1/np.median(signal[:int(len(signal)/6)])
    #change point detection for a signal
    algo = rpt.KernelCPD(kernel="rbf", params={"gamma": gamma}, min_size=10).fit(signal=signal)
    bkps_pred = algo.predict(pen=beta*np.log(signal.shape[0])) #use PELT with \pen = Beta*log(T)
    return bkps_pred

def get_list_of_bkps_pred(list_of_signals, beta = 0.4, gamma = None):
    if gamma is None: 
        gamma = 1/np.median(list_of_signals[0][:int(len(list_of_signals[0])/6)])
    #change point detection for a signal divided in a list of shorter sub-signals (in parallel)
    get_bkps_pred_delayed = delayed(lambda s: get_bkps_pred(s, beta=beta, gamma = gamma))
    list_of_bkps_pred = Parallel(n_jobs=-2)(get_bkps_pred_delayed(s) for s in list_of_signals)
    return list_of_bkps_pred
