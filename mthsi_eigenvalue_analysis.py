#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# custom code imports
import utils.read as read
import utils.eigencomp as eigencomp

#%%
def sort_mthsi_df(df):
    date_list = df['Date'].unique()
    vals_list = []
    for date in date_list:
        filter = df['Date'] == date
        vals = np.array(df[filter]['Eigenvalues'])
        vals_list += [vals]
    return {'Dates': date_list, 'Eigenvalues': np.array(vals_list)}


#%% load in data
veg_isomap_eigenvalues = pd.read_csv('veg_isomap_eigenvalues.csv')
veg_laplacian_eigenvalues = pd.read_csv('veg_laplacian_eigenvalues.csv')
veg_lle_eigenvalues = pd.read_csv('veg_lle_eigenvalues.csv')
veg_pca_eigenvalues = pd.read_csv('veg_pca_eigenvalues.csv')
# %% sort data
veg_isomap_dict = sort_mthsi_df(veg_isomap_eigenvalues)
veg_laplacian_dict = sort_mthsi_df(veg_laplacian_eigenvalues)
veg_lle_dict = sort_mthsi_df(veg_lle_eigenvalues)
veg_pca_dict = sort_mthsi_df(veg_pca_eigenvalues)

# %%
# for veg, use all data; for city, [1::]
veg_isomap_eigens = veg_isomap_dict['Eigenvalues']
veg_laplacian_eigens = veg_laplacian_dict['Eigenvalues']
veg_lle_eigens = veg_lle_dict['Eigenvalues']
veg_pca_eigens = veg_pca_dict['Eigenvalues']

#%%
mode = 'diff'

# for veg scenes
match mode:
    case 'diff':
        time = [2,4,6,8]
    case 'start':
        time = [0,2,4,6,8]
    case 'end':
        time = [0,2,4,6,8]

# for city scenes
"""
match mode:
    case 'diff':
        time = [2,4,6,8]
    case 'start':
        time = [0,2,4,6,8]
    case 'end':
        time = [0,2,4,6,8]
"""

print(time)

for metric in ['sam','euclidean','manhattan']:
    iso_metric = eigencomp.get_metrics_from_list(veg_isomap_eigens,metric=metric,mode=mode)
    lap_metric = eigencomp.get_metrics_from_list(veg_laplacian_eigens,metric=metric,mode=mode)
    lle_metric = eigencomp.get_metrics_from_list(veg_lle_eigens,metric=metric,mode=mode)
    pca_metric = eigencomp.get_metrics_from_list(veg_pca_eigens,metric=metric,mode=mode)

    plt.figure()
    plt.plot(time,iso_metric,label='ISOMAP',marker='o')
    plt.plot(time,lap_metric,label='Laplacian Eigenmaps',marker='o')
    plt.plot(time,lle_metric,label='LLE',marker='o')
    plt.plot(time,pca_metric,label='PCA',marker='o')
    plt.xlabel('Time [weeks]')
    plt.ylabel(f'Change from {mode} of {metric} distance')
    plt.legend()


# %%

time_dict = {
    'ISOMAP': veg_isomap_eigens,
    'Laplacian Eigenmaps': veg_laplacian_eigens,
    'LLE': veg_lle_eigens,
    'PCA': veg_pca_eigens,

}

labels = ['Week 0','Week 2','Week 4','Week 6','Week 8']

for emb in [*time_dict.keys()]:
    time = [0,2,4,6,8]
    plt.figure()
    i = 0
    for eigens in time_dict[emb]:
        plt.plot(eigens,label=labels[i])
        i+=1
    if emb != 'LLE':
        plt.yscale('log')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue Magnitude')
    plt.title(f"{emb} Eigenvalues")
    plt.legend()

# %%
