#%%
from scipy.io import savemat, loadmat
import numpy as np


L1 = loadmat('DD-45left01-SPL.mat')['data'][0, 0]
L2 = loadmat('DD-45left02-SPL.mat')['data'][0, 0]
L3 = loadmat('DD-45left03-SPL.mat')['data'][0, 0]
R1 = loadmat('DD-45right01-SPL.mat')['data'][0, 0]
R2 = loadmat('DD-45right02-SPL.mat')['data'][0, 0]
R3 = loadmat('DD-45right03-SPL.mat')['data'][0, 0]

LD1 = {name: L1[name] for name in L1.dtype.names}
LD2 = {name: L2[name] for name in L2.dtype.names}
LD3 = {name: L3[name] for name in L3.dtype.names}
RD1 = {name: R1[name] for name in R1.dtype.names}
RD2 = {name: R2[name] for name in R2.dtype.names}
RD3 = {name: R3[name] for name in R3.dtype.names}

fxA = LD1['fx'].squeeze()
HL1 = np.abs(LD1['TrFuncAE'].squeeze())
HL2 = np.abs(LD2['TrFuncAE'].squeeze())
HL3 = np.abs(LD3['TrFuncAE'].squeeze())
HR1 = np.abs(RD1['TrFuncAE'].squeeze())
HR2 = np.abs(RD2['TrFuncAE'].squeeze())
HR3 = np.abs(RD3['TrFuncAE'].squeeze())

print(fxA, HL1, HL2, HL3, HR1, HR2, HR3)

HLrms = np.sqrt((HL1**2 + HL2**2 + HL3**2) / 3)
HRrms = np.sqrt((HR1**2 + HR2**2 + HR3**2) / 3)

print(fxA, HLrms, HRrms)
# %%
for chan in ['L', 'R']:
    results = {}
    results["headName"] = "DD-45"
    results["fx"] = []
    results["TrFuncAE"] = []
    if chan == 'L':
        results['Channel'] = 'L'
        for f in fxA:
            idx = np.where(fxA == f)
            results["fx"].append(f)
            results["TrFuncAE"].append(HLrms[idx][0])
        savemat("DD-45left-rms.mat", {'data': results})
    else:
        results['Channel'] = 'R'
        for f in fxA:
            idx = np.where(fxA == f)
            results["fx"].append(f)
            results["TrFuncAE"].append(HRrms[idx][0])
        savemat("DD-45right-rms.mat", {'data': results})
#%%