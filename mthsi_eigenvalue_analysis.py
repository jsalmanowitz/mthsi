#%% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
veg_isomap_eigenvalues = pd.read_csv('city_isomap_eigenvalues.csv')
veg_laplacian_eigenvalues = pd.read_csv('city_laplacian_eigenvalues.csv')
veg_lle_eigenvalues = pd.read_csv('city_lle_eigenvalues.csv')
veg_pca_eigenvalues = pd.read_csv('city_pca_eigenvalues.csv')
# %% sort data
veg_isomap_dict = sort_mthsi_df(veg_isomap_eigenvalues)
veg_laplacian_dict = sort_mthsi_df(veg_laplacian_eigenvalues)
veg_lle_dict = sort_mthsi_df(veg_lle_eigenvalues)
veg_pca_dict = sort_mthsi_df(veg_pca_eigenvalues)

#%% functions to move later

def get_metrics_from_list(eigen_list,metric):

    def get_cos_sim(eigen_list_left,eigen_list_right):
        cos_vals = []
        for index in range(len(eigen_list_left)):
            vec1 = eigen_list_left[index]
            vec2 = eigen_list_right[index]
            cos_sim = np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            cos_vals += [cos_sim]
        return np.array(cos_vals[1:-1])
    
    def get_sam(eigen_list_left,eigen_list_right):
        cos_sim = get_cos_sim(eigen_list_left,eigen_list_right)
        return np.arccos(cos_sim)*180/np.pi
    
    def get_euclidean(eigen_list_left,eigen_list_right):
        diff = eigen_list_left-eigen_list_right
        return np.linalg.norm(diff,axis=1)[1:-1]
    
    def get_manhattan(eigen_list_left,eigen_list_right):
        diff = eigen_list_left-eigen_list_right
        return np.sum(abs(diff),axis=1)[1:-1]

    nan_array = np.full(len(eigen_list[0]), np.nan)
    nan_array = nan_array[np.newaxis,:]
    eigen_list_left = np.concatenate([nan_array,eigen_list])
    eigen_list_right = np.concatenate([eigen_list,nan_array])

    if metric == 'cosine':
        return get_cos_sim(eigen_list_left,eigen_list_right)
    
    elif metric == 'sam':
        return get_sam(eigen_list_left,eigen_list_right)
    
    elif metric == 'euclidean':
        return get_euclidean(eigen_list_left,eigen_list_right)
    
    elif metric == 'manhattan':
        return get_manhattan(eigen_list_left,eigen_list_right)

    
# %%
veg_isomap_eigens = veg_isomap_dict['Eigenvalues'][1::]
veg_laplacian_eigens = veg_laplacian_dict['Eigenvalues'][1::]
veg_lle_eigens = veg_lle_dict['Eigenvalues'][1::]
veg_pca_eigens = veg_pca_dict['Eigenvalues'][1::]

for metric in ['sam','euclidean','manhattan']:
    iso_metric = get_metrics_from_list(veg_isomap_eigens,metric=metric)
    lap_metric = get_metrics_from_list(veg_laplacian_eigens,metric=metric)
    lle_metric = get_metrics_from_list(veg_lle_eigens,metric=metric)
    pca_metric = get_metrics_from_list(veg_pca_eigens,metric=metric)
    time = [0,2,4,6]

    plt.figure()
    plt.plot(time,iso_metric,label='ISOMAP',marker='o')
    plt.plot(time,lap_metric,label='Laplacian Eigenmaps',marker='o')
    plt.plot(time,lle_metric,label='LLE',marker='o')
    plt.plot(time,pca_metric,label='PCA',marker='o')
    plt.xlabel('Time [weeks]')
    plt.ylabel(f'Change in {metric} distance')
    plt.legend()

    plt.figure()
    plt.plot(time,iso_metric/iso_metric[0],label='ISOMAP',marker='o')
    plt.plot(time,lap_metric/lap_metric[0],label='Laplacian Eigenmaps',marker='o')
    plt.plot(time,lle_metric/lle_metric[0],label='LLE',marker='o')
    plt.plot(time,pca_metric/pca_metric[0],label='PCA',marker='o')
    plt.xlabel('Time [weeks]')
    plt.ylabel(f'Normalized Change in {metric} distance')
    plt.legend()

# %%
