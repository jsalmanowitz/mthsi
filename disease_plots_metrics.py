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
file_path_healthy = "/home/js9002/mthsi/*healthy_plot_batch_isos*.npy"
healthy_files = sorted(glob.glob(file_path_healthy))
file_path_healthy_diff  = "/home/js9002/mthsi/*healthy_plot_diff*.npy"
healthy_diff_files = sorted(glob.glob(file_path_healthy_diff))
#%% prepare data

index = 4

baseline = np.load(healthy_files[index])[1,1,:]
baseline = baseline/np.linalg.norm(baseline)

disease_severity = []
for plot in disease_severity_files:
    sev = np.load(plot)
    sev_ind = sev[index]
    disease_severity += [sev_ind]

ind_eigens = []
for plot in disease_files:
    eigens = np.load(plot)
    eigens_ind = eigens[index]/np.linalg.norm(eigens[index])
    ind_eigens += [eigens_ind]

eigen_diff = eigencomp.get_metrics_from_list(ind_eigens,'euclidean',mode='compare',compare_list=baseline)

#%% generate stats for healthy data

from scipy.optimize import curve_fit
from scipy.special import gamma

eigens_h = np.load(healthy_files[index])
diffs = np.linalg.norm(eigens_h[:,0,:] - eigens_h[:,1,:],axis=1)

mean_diff = np.mean(diffs)
std_diff = np.std(diffs)

counts, edges = np.histogram(diffs, bins=20)
bin_centers = (edges[:-1] + edges[1:]) / 2

def folded_gaussian(x,mean,std,a):
    numerator_1 = np.exp(-(x-mean)**2/(2*std**2))
    numerator_2 = np.exp(-(x+mean)**2/(2*std**2))
    denominator = std*np.sqrt(2*np.pi)
    return a*(numerator_1+numerator_2)/denominator

initial_guess = [0.02, 0.05, 10]
popt, pcov = curve_fit(folded_gaussian, bin_centers, counts, p0=initial_guess)
print(popt)
print(pcov)

popt_healthy = popt

plt.figure()
plt.hist(diffs, bins=20,label='measured')
plt.plot(bin_centers, 
         folded_gaussian(bin_centers,popt[0],popt[1],popt[2]),
         label='model')
plt.legend()

#%%

unhealthy_eigen = ind_eigens[0]
compare_eigens = []
for i in range(len(eigens_h)):
    compare_eigens += [unhealthy_eigen]
compare_eigens = np.array(compare_eigens)

diffs_unhealthy = np.linalg.norm(eigens_h[:,1,:]-compare_eigens,axis=1)

counts, edges = np.histogram(diffs_unhealthy, bins=20)
bin_centers = (edges[:-1] + edges[1:]) / 2

initial_guess = [0.1, 0.05, 10]
popt, pcov = curve_fit(folded_gaussian, bin_centers, counts, p0=initial_guess)
print(popt)
print(pcov)

plt.figure()
plt.hist(diffs_unhealthy, bins=30,label='measured')
plt.plot(bin_centers, 
         folded_gaussian(bin_centers,popt[0],popt[1],popt[2]),
         label='model')
plt.legend()

#%% batch stats for unhealthy eigenvalues

mean_list = []
std_list = []

j = 0
for unhealthy_eigen in ind_eigens:
    compare_eigens = []
    for i in range(len(eigens_h)):
        compare_eigens += [unhealthy_eigen]
    compare_eigens = np.array(compare_eigens)
    diffs_unhealthy = np.linalg.norm(eigens_h[:,1,:]-compare_eigens,axis=1)
    counts, edges = np.histogram(diffs_unhealthy, bins=20)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    initial_guess = [0.14, 0.03, 20]
    popt, pcov = curve_fit(folded_gaussian, bin_centers, counts, p0=initial_guess,maxfev=1000)
    mean_list += [abs(popt[0])]
    std_list += [abs(popt[1])]
    j += 1

std_list = np.array(std_list)
#%%
plt.figure()
plt.scatter(disease_severity,eigen_diff)
plt.axhline(popt_healthy[0],label='healthy mean',color='red')
plt.axhline(popt_healthy[0]+2*popt_healthy[1],label='healthy 95% CI',color='blue')
plt.xlabel('Disease Severity %')
plt.ylabel('Euclidean Distance')
plt.legend()

# %%

plt.figure()
plt.errorbar(disease_severity, mean_list, yerr=2*std_list, fmt='o', capsize=5)
plt.axhline(popt_healthy[0],label='healthy mean',color='red')
plt.axhline(popt_healthy[0]+2*popt_healthy[1],label='healthy 95% CI',color='blue')
plt.xlabel('Disease Severity %')
plt.ylabel('Euclidean Distance')
plt.legend()

# %%
