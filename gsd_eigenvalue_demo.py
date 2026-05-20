#%% imports

import numpy as np
import pandas as pd
import spectral.io.envi as envi
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.manifold import Isomap
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
from scipy.linalg import eigh

import utils
#%% utils for read-in and plotting

def getHyperspectralCube(sensor,file_path,file_name,header_name=None):
    # reads in hyperspectral cube with associated metadata

    if sensor == 'spectir':
        if header_name is not None:
            img = envi.open(file_path+file_name,file_path+header_name)
            data = img.load()
            metadata = img.metadata
            wavelengths = np.array(img.bands.centers)*1000
            data_numpy = np.array(data)

            return{
                'cube': data_numpy,
                'wavelengths': wavelengths,
                'file_name': file_name,
                'sensor': sensor,
                'metadata': metadata
            }
        else:
            print('Header File Path (.dat) not specified. Returning None.')
            return None

def getRGB(cube,wavelengths):

    wvl = wavelengths

    rIndex = ((wvl-650) > 0).argmax()
    gIndex = ((wvl-550) > 0).argmax()
    bIndex = ((wvl-450) > 0).argmax()

    rData = cube[:,:,rIndex]
    gData = cube[:,:,gIndex]
    bData = cube[:,:,bIndex]

    colorData = np.array([rData,gData,bData])
    # normalize
    colorMax = np.max(np.concatenate(np.concatenate(colorData)))
    colorImage = colorData/colorMax

    return colorImage.swapaxes(0,2)

def plotRGB(cubeData,use_plotly = False, use_cube = False):
    
    colorImage = getRGB(cubeData['cube'],cubeData['wavelengths'])
    file_name = cubeData['file_name']

    if use_plotly:
        fig = px.imshow(colorImage,title=file_name)

        fig.show()
    else:
        fig = plt.figure()
        plt.imshow(colorImage)
        plt.title(file_name)
        plt.show()
    
    return

def downsample_avg(arr, factor):
    # generated using Goolge Gemini
    sh = arr.shape
    # Reshape to (new_ax0, factor, new_ax1, factor, depth)
    reshaped = arr.reshape(sh[0]//factor, factor, sh[1]//factor, factor, sh[2])
    # Mean across the factor axes (1 and 3)
    return reshaped.mean(axis=(1, 3))

#%% utils for graph analysis

def get_laplacian_evs(X, k, num):
        # Build normalized Laplacian
        A = kneighbors_graph(X, k, mode='distance',metric='cosine',include_self=True)
        A = 0.5 * (A + A.T)
        L = laplacian(A, normed=True)
        # Get smallest k+1 eigenvalues (1st is always 0)
        vals, _ = eigsh(L, k=num+1, which='SM')
        return vals[1:] # Skip the trivial zero

def get_lle_evs(X, k, num):
    # Generated using Google Gemini

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


#%%
# use 1631 for veg?
# use 1559 for industrial
file_path = r"./data/SHARE2012/"
file_name = r"0920-1559_pol_ref.hdr"
header_name = r"0920-1559_pol_ref.dat"

spectirCube = getHyperspectralCube('spectir',file_path,file_name,header_name)

plotRGB(spectirCube,use_plotly=False)

#%%
# cube for vegetation area
# subset_1m = spectirCube['cube'][3300:3492,120:-8,:]
# cube for city area
subset_1m = spectirCube['cube'][1000:1192,120:-8,:]
subset_2m = downsample_avg(subset_1m,factor=2)
print(np.shape(subset_2m))
subset_4m = downsample_avg(subset_2m,factor=2)
print(np.shape(subset_4m))
subset_8m = downsample_avg(subset_4m,factor=2)
print(np.shape(subset_8m))
subset_16m = downsample_avg(subset_8m,factor=2)
print(np.shape(subset_16m))
subset_32m = downsample_avg(subset_16m,factor=2)
print(np.shape(subset_32m))

subsetRGB_1m = getRGB(subset_1m,spectirCube['wavelengths'])
subsetRGB_2m = getRGB(subset_2m,spectirCube['wavelengths'])
subsetRGB_4m = getRGB(subset_4m,spectirCube['wavelengths'])
subsetRGB_8m = getRGB(subset_8m,spectirCube['wavelengths'])
subsetRGB_16m = getRGB(subset_16m,spectirCube['wavelengths'])
subsetRGB_32m = getRGB(subset_32m,spectirCube['wavelengths'])

#%%
fig = plt.figure()
plt.imshow(subsetRGB_1m)
plt.title('SpecTIR Image - 1m GSD')
plt.show()

fig = plt.figure()
plt.imshow(subsetRGB_2m)
plt.title('SpecTIR Image - 2m GSD')
plt.show()

fig = plt.figure()
plt.imshow(subsetRGB_4m)
plt.title('SpecTIR Image - 4m GSD')
plt.show()

fig = plt.figure()
plt.imshow(subsetRGB_8m)
plt.title('SpecTIR Image - 8m GSD')
plt.show()

fig = plt.figure()
plt.imshow(subsetRGB_16m)
plt.title('SpecTIR Image - 16m GSD')
plt.show()

fig = plt.figure()
plt.imshow(subsetRGB_32m)
plt.title('SpecTIR Image - 32m GSD')
plt.show()

# %% Set up data for batch processing

gsd_data = {
    '1m': subset_1m,
    '2m': subset_2m,
    '4m': subset_4m,
    '8m': subset_8m,
    '16m': subset_16m,
    '32m': subset_32m
}

# %% Laplacian implementation test bed
start = datetime.now()
print(start)

subset_pix = np.concatenate(subset_32m)

ev_spec = get_laplacian_evs(subset_pix,34,int(np.sqrt(len(data_concat))))

end = datetime.now()
print(end)
print(f"The total elapsed time was {end-start}")

#%% Laplacian eigenvalues to CSV unscaled
gsd_eigendata = pd.DataFrame(columns=['GSD','Eigenvalues'])

for gsd in [*gsd_data.keys()]:
    data_concat = np.concatenate(gsd_data[gsd])
    ev_spec = get_laplacian_evs(data_concat,34,34)

    ev_df = pd.DataFrame({'GSD': gsd, 'Eigenvalues': ev_spec})

    gsd_eigendata = pd.concat([gsd_eigendata, ev_df], ignore_index=True)
gsd_eigendata.to_csv('gsd_eigenvalues_laplacian_city.csv', index=False)

#%% Laplacian eigenvalues - scaled 1 - scaled num
gsd_eigendata = pd.DataFrame(columns=['GSD','Eigenvalues'])

for gsd in [*gsd_data.keys()]:
    data_concat = np.concatenate(gsd_data[gsd])
    ev_spec = get_laplacian_evs(data_concat,34,int(np.sqrt(len(data_concat))))

    ev_df = pd.DataFrame({'GSD': gsd, 'Eigenvalues': ev_spec})

    gsd_eigendata = pd.concat([gsd_eigendata, ev_df], ignore_index=True)
gsd_eigendata.to_csv('gsd_eigenvalues_laplacian_city_scaled_1.csv', index=False)

#%% Laplacian eigenvalues - scaled 2 - scaled k and num
gsd_eigendata = pd.DataFrame(columns=['GSD','Eigenvalues'])

for gsd in [*gsd_data.keys()]:
    data_concat = np.concatenate(gsd_data[gsd])
    ev_spec = get_laplacian_evs(data_concat,int(np.sqrt(len(data_concat))),int(np.sqrt(len(data_concat))))

    ev_df = pd.DataFrame({'GSD': gsd, 'Eigenvalues': ev_spec})

    gsd_eigendata = pd.concat([gsd_eigendata, ev_df], ignore_index=True)
gsd_eigendata.to_csv('gsd_eigenvalues_laplacian_city_scaled_2.csv', index=False)

# %%

gsd_laplacian_df = pd.read_csv('gsd_eigenvalues_laplacian_city_scaled_1.csv')
gsd_series = gsd_laplacian_df['GSD']
gsds = gsd_series.unique()

plt.figure()
for gsd in gsds:
    filter = gsd_laplacian_df['GSD'] == gsd
    gsd_vals = gsd_laplacian_df[filter]['Eigenvalues']
    plt.plot(np.linspace(0,1,len(gsd_vals)),gsd_vals,label = gsd)
    print(np.sum(gsd_vals)/len(gsd_vals))
plt.legend()
plt.xlabel("Normalized Eigenvalue Index (k)")
plt.ylabel("Eigenvalue Magnitude ($\lambda$)")

# %% ISOMAP implementation test bed
start = datetime.now()
print(start)

subset_pix = np.concatenate(subset_2m)

iso = Isomap(n_components=34,n_neighbors=34,metric='cosine')
iso.fit(subset_pix)
ev_spec = iso.kernel_pca_.eigenvalues_
ev_spec_norm = ev_spec/np.sum(ev_spec)
print(np.sum(ev_spec_norm))

end = datetime.now()
print(end)
print(f"The total elapsed time was {end-start}")
#%% ISOMAP implementation batch - unscaled

gsd_eigendata = pd.DataFrame(columns=['GSD','Eigenvalues'])

for gsd in [*gsd_data.keys()][1::]:
    data_concat = np.concatenate(gsd_data[gsd])
    iso = Isomap(n_components=34,n_neighbors=34,metric='cosine')
    iso.fit(data_concat)
    ev_spec = iso.kernel_pca_.eigenvalues_
    ev_spec_norm = ev_spec/np.sum(ev_spec)
    print(np.sum(ev_spec_norm))

    ev_df = pd.DataFrame({'GSD': gsd, 'Eigenvalues': ev_spec_norm})

    gsd_eigendata = pd.concat([gsd_eigendata, ev_df], ignore_index=True)
gsd_eigendata.to_csv('gsd_eigenvalues_city_isomap.csv', index=False)
#%% ISOMAP implementation batch - scaled 1 - scaled num

gsd_eigendata = pd.DataFrame(columns=['GSD','Eigenvalues'])

for gsd in [*gsd_data.keys()][1::]:
    data_concat = np.concatenate(gsd_data[gsd])
    iso = Isomap(n_components=int(np.sqrt(len(data_concat))),n_neighbors=34,metric='cosine')
    iso.fit(data_concat)
    ev_spec = iso.kernel_pca_.eigenvalues_
    ev_spec_norm = ev_spec/np.sum(ev_spec)

    ev_df = pd.DataFrame({'GSD': gsd, 'Eigenvalues': ev_spec_norm})

    gsd_eigendata = pd.concat([gsd_eigendata, ev_df], ignore_index=True)
    
gsd_eigendata.to_csv('gsd_eigenvalues_isomap_city_scaled_1.csv', index=False)

#%% ISOMAP implementation batch - scaled 2 - scaled k and num

gsd_eigendata = pd.DataFrame(columns=['GSD','Eigenvalues'])

for gsd in [*gsd_data.keys()][1::]:
    data_concat = np.concatenate(gsd_data[gsd])
    iso = Isomap(n_components=int(np.sqrt(len(data_concat))),n_neighbors=int(np.sqrt(len(data_concat))),metric='cosine')
    iso.fit(data_concat)
    ev_spec = iso.kernel_pca_.eigenvalues_
    ev_spec_norm = ev_spec/np.sum(ev_spec)

    ev_df = pd.DataFrame({'GSD': gsd, 'Eigenvalues': ev_spec_norm})

    gsd_eigendata = pd.concat([gsd_eigendata, ev_df], ignore_index=True)
    
gsd_eigendata.to_csv('gsd_eigenvalues_isomap_city_scaled_2.csv', index=False)

# %% Plot isomap data

gsd_laplacian_df = pd.read_csv('gsd_eigenvalues_isomap_city_scaled_2.csv')
gsd_series = gsd_laplacian_df['GSD']
gsds = gsd_series.unique()

plt.figure()
for gsd in gsds:
    filter = gsd_laplacian_df['GSD'] == gsd
    gsd_vals = gsd_laplacian_df[filter]['Eigenvalues']
    plt.semilogy(np.linspace(0,1,len(gsd_vals)),
                 gsd_vals,label = gsd)

plt.legend()
plt.xlabel("Normalized Eigenvalue Index (k)")
plt.ylabel("Eigenvalue Magnitude ($\lambda$)")

# %% LLE implementation test bed
start = datetime.now()
print(start)

subset_pix = np.concatenate(subset_2m)

ev_spec = get_lle_evs(subset_pix,35,34)

end = datetime.now()
print(end)
print(f"The total elapsed time was {end-start}")

#%% LLE implementation batch - unscaled

gsd_eigendata = pd.DataFrame(columns=['GSD','Eigenvalues'])

for gsd in [*gsd_data.keys()][1::]:
    data_concat = np.concatenate(gsd_data[gsd])
    ev_spec = get_lle_evs(data_concat,35,34)

    ev_df = pd.DataFrame({'GSD': gsd, 'Eigenvalues': ev_spec})

    gsd_eigendata = pd.concat([gsd_eigendata, ev_df], ignore_index=True)
gsd_eigendata.to_csv('gsd_eigenvalues_lle.csv', index=False)

#%% LLE implementation batch - scaled 1 - scaled num

gsd_eigendata = pd.DataFrame(columns=['GSD','Eigenvalues'])

for gsd in [*gsd_data.keys()][1::]:
    data_concat = np.concatenate(gsd_data[gsd])
    ev_spec = get_lle_evs(data_concat,34,int(np.sqrt(len(data_concat))))

    ev_df = pd.DataFrame({'GSD': gsd, 'Eigenvalues': ev_spec})

    gsd_eigendata = pd.concat([gsd_eigendata, ev_df], ignore_index=True)
gsd_eigendata.to_csv('gsd_eigenvalues_lle_city_scaled_1.csv', index=False)


#%% LLE implementation batch - scaled 2 - scaled k and num

gsd_eigendata = pd.DataFrame(columns=['GSD','Eigenvalues'])

for gsd in [*gsd_data.keys()][1::]:
    data_concat = np.concatenate(gsd_data[gsd])
    ev_spec = get_lle_evs(data_concat,int(np.sqrt(len(data_concat))),int(np.sqrt(len(data_concat))))

    ev_df = pd.DataFrame({'GSD': gsd, 'Eigenvalues': ev_spec})

    gsd_eigendata = pd.concat([gsd_eigendata, ev_df], ignore_index=True)
gsd_eigendata.to_csv('gsd_eigenvalues_lle_city_scaled_2.csv', index=False)

# %% Plot lle data

gsd_laplacian_df = pd.read_csv('gsd_eigenvalues_lle_city_scaled_2.csv')
gsd_series = gsd_laplacian_df['GSD']
gsds = gsd_series.unique()

plt.figure()
for gsd in gsds:
    filter = gsd_laplacian_df['GSD'] == gsd
    gsd_vals = gsd_laplacian_df[filter]['Eigenvalues']
    plt.plot(np.linspace(0,1,len(gsd_vals)),
                 gsd_vals,label = gsd)

plt.legend()
plt.xlabel("Eigenvalue Index (k)")
plt.ylabel("Eigenvalue Magnitude ($\lambda$)")
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veg_df = pd.read_csv('gsd_eigenvalues_lle_scaled_1.csv')
city_df = pd.read_csv('gsd_eigenvalues_lle_city_scaled_1.csv')

diff_df = pd.DataFrame({
    'GSD': veg_df['GSD'].copy(),
    'Eigenvalues': veg_df['Eigenvalues']-city_df['Eigenvalues']
})
mags = []
angles = []

gsds = ['1m','2m','4m','8m','16m','32m']

for gsd in gsds:
    filter = diff_df['GSD'] == gsd
    gsd_vals = diff_df[filter]['Eigenvalues']

    gsd_mag = np.linalg.norm(gsd_vals)

    veg_vals = np.array(veg_df[filter]['Eigenvalues'])
    city_vals =np.array(city_df[filter]['Eigenvalues'])

    cos_sim = np.dot(veg_vals,city_vals)/(np.linalg.norm(veg_vals)*np.linalg.norm(city_vals))
    angle = np.arccos(cos_sim)
    angle_deg = 180/np.pi*angle
    
    mags += [gsd_mag]
    angles += [angle_deg]

print(angles)

#%%

gsd = [1,2,4,8,16,32]
gsd_2 = gsd[1::]

plt.figure()
plt.plot(gsd,mags)
plt.xlabel('GSD [m]')
plt.ylabel('Euclidean Distance')

plt.figure()
plt.plot(gsd,angles)
plt.xlabel('GSD [m]')
plt.ylabel('Angle [deg]')


# %%
