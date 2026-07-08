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
# %%
