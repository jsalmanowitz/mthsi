#%%
import numpy as np
from utils import eigencomp
import matplotlib.pyplot as plt

#%%

isos_1 = np.load('disease_plot_isos_plot_001.npy')

isos_1_rescaled = []
for isos in isos_1:
    iso_scaled = isos/np.linalg.norm(isos)
    isos_1_rescaled += [iso_scaled]

diff_list = eigencomp.get_metrics_from_list(np.log10(isos_1_rescaled),'manhattan',mode='diff')
start_list = eigencomp.get_metrics_from_list(np.log10(isos_1_rescaled),'manhattan',mode='start')

plt.figure()
plt.plot(diff_list,label='diff')
plt.plot(start_list,label='start')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Difference Metric')
plt.title('Plot 005 Manhattan Difference Over Time')
# %% get saved disease data
import glob
import numpy as np
file_path = "/home/js9002/mthsi/*disease*.npy"
disease_files = sorted(glob.glob(file_path))

d5_eigens = []
for plot in disease_files:
    eigens = np.load(plot)
    eigens_D5 = eigens[4]
    d5_eigens += [eigens_D5]




# %%
