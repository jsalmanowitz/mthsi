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
from utils import eigencomp
import matplotlib.pyplot as plt

file_path = "/home/js9002/mthsi/*disease_plot_isos*.npy"
disease_files = sorted(glob.glob(file_path))
file_path_sev = "/home/js9002/mthsi/*disease_severity*.npy"
disease_severity_files = sorted(glob.glob(file_path_sev))
file_path_healthy = "/home/js9002/mthsi/*healthy_plot_isos*.npy"
healthy_files = sorted(glob.glob(file_path_healthy))
file_path_healthy_diff  = "/home/js9002/mthsi/*healthy_plot_diff*.npy"
healthy_diff_files = sorted(glob.glob(file_path_healthy_diff))
#%% prepare data
d5_baseline = np.load(healthy_files[0])
d5_baseline = d5_baseline/np.linalg.norm(d5_baseline)

d5_disease_severity = []
for plot in disease_severity_files:
    sev = np.load(plot)
    sev_d5 = sev[4]
    d5_disease_severity += [sev_d5]

d5_eigens = []
for plot in disease_files:
    eigens = np.load(plot)
    eigens_D5 = eigens[4]/np.linalg.norm(eigens[4])
    d5_eigens += [eigens_D5]

eigen_diff = eigencomp.get_metrics_from_list(d5_eigens,'euclidean',mode='compare',compare_list=d5_baseline)

#%% generate stats

diffs = np.load(healthy_diff_files[0])
mean_diff = np.mean(diffs)
std_diff = np.std(diffs)

plt.figure()
plt.hist(diffs, bins=30)


#%%
plt.figure()
plt.scatter(d5_disease_severity,eigen_diff)
plt.axhline(mean_diff,label='healthy mean',color='red')
plt.axhline(mean_diff+2*std_diff,label='healthy 95% CI',color='blue')
plt.xlabel('Disease Severity %')
plt.ylabel('Euclidean Distance')
plt.legend()

# %%
