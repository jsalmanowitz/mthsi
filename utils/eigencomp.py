#%% imports

import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.manifold import Isomap
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
from scipy.linalg import eigh
from sklearn.decomposition import PCA

#%% utils for graph analysis
def clean_cube(hyperCube):
     # removes masked portions of hyperspectral cube (pixels with no data)
     X = np.concatenate(hyperCube['cube'])
     X = X[np.where(np.linalg.norm(X,axis=1) > 0)]
     return X

def sort_mthsi_df(df):
    # sorts mthsi dataframe into dictionary for easy use
    # returns dict which containts list of dates and corresponding list of eigenvalue arrays
    date_list = df['Date'].unique()
    vals_list = []
    for date in date_list:
        filter = df['Date'] == date
        vals = np.array(df[filter]['Eigenvalues'])
        vals_list += [vals]
    return {'Dates': date_list, 'Eigenvalues': np.array(vals_list)}

def get_metrics_from_list(eigen_list,metric,mode='diff'):

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

    if mode == 'diff':
        nan_array = np.full(len(eigen_list[0]), np.nan)
        nan_array = nan_array[np.newaxis,:]
        eigen_list_left = np.concatenate([nan_array,eigen_list])
        eigen_list_right = np.concatenate([eigen_list,nan_array])
    elif mode == 'start':
        start_array = np.full(len(eigen_list[0]), eigen_list[0])
        start_array = start_array[np.newaxis,:]
        eigen_list_left = np.concatenate([start_array,eigen_list])
        eigen_list_right = start_array
        for i in range(len(eigen_list)):
            eigen_list_right = np.concatenate([eigen_list_right,start_array])

    elif mode == 'end':
        end_array = np.full(len(eigen_list[0]), eigen_list[0])
        end_array = start_array[np.newaxis,:]
        eigen_list_left = np.concatenate([end_array,eigen_list])
        eigen_list_right = end_array
        for i in range(len(eigen_list[1])):
            eigen_list_right = np.concatenate([eigen_list_right,end_array])

    if metric == 'cosine':
        return get_cos_sim(eigen_list_left,eigen_list_right)
    
    elif metric == 'sam':
        return get_sam(eigen_list_left,eigen_list_right)
    
    elif metric == 'euclidean':
        return get_euclidean(eigen_list_left,eigen_list_right)
    
    elif metric == 'manhattan':
        return get_manhattan(eigen_list_left,eigen_list_right)
    
def get_metric_from_dict(dictionary,metric):
    eigenvalue_list = dictionary['Eigenvalues']
    return get_metrics_from_list(eigenvalue_list)
#%% generate eigenvalue distributions

def get_pca_evs(hyperCube,num):
    X = clean_cube(hyperCube)
    pca = PCA(n_components=num)
    pca.fit(X)
    ev_spec = pca.explained_variance_
    ev_spec_norm = ev_spec/np.sum(ev_spec)
    return ev_spec_norm

def get_laplacian_evs(hyperCube, k, num, mode='distance',metric='cosine'):
        # Portions generated using Google Gemini
        X = clean_cube(hyperCube)
        # Build normalized Laplacian
        A = kneighbors_graph(X, k, mode=mode,metric=metric,include_self=True)
        A = 0.5 * (A + A.T)
        L = laplacian(A, normed=True)
        # Get smallest k+1 eigenvalues (1st is always 0)
        vals, _ = eigsh(L, k=num+1, which='SM')
        return vals[1:] # Skip the trivial zero

def get_lle_evs(hyperCube, k, num):
    # Generated using Google Gemini
    X = clean_cube(hyperCube)
    # 1. Compute the weight matrix W
    W = barycenter_kneighbors_graph(X, n_neighbors=k)

    # 2. Construct the M matrix: M = (I - W)^T * (I - W)
    # This is the cost matrix that LLE minimizes
    import scipy.sparse as sp
    I = sp.eye(W.shape[0])
    M = (I - W).T @ (I - W)

    # 3. Get the eigenvalues
    # If X is small, use eigh (dense); if X is large, use eigsh (sparse)
    eigenvalues = eigh(M.toarray(), eigvals_only=True)
    return eigenvalues[1:num+1]

def get_iso_evs(hyperCube,k,num,metric='cosine'):
    X = clean_cube(hyperCube)
    iso = Isomap(n_components=num,n_neighbors=k,metric=metric)
    iso.fit(X)
    ev_spec = iso.kernel_pca_.eigenvalues_
    ev_spec_norm = ev_spec/np.sum(ev_spec)
    return ev_spec_norm