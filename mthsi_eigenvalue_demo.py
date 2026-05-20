#%% utils
# package imports
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
# custom code imports
import utils.read as read
import utils.eigencomp as eigencomp

#%% read-in file names for test read
shots_path = r"./data/tanager/lake_shots/"
tanager_files = read.getFiles(shots_path)

# %% test read in files to find crops
file = tanager_files[1]
cubeTest = read.getHyperspectralCube('tanager',shots_path,fr"{file}")
slicedCubeTest = read.sliceCube(cubeTest,[74,94,455,485])
# %% test plot cubes to find crops

read.plotRGB(cubeTest,use_plotly=True)
read.plotRGB(slicedCubeTest)


#%% Automated read-in of all files for lake shots
lake_shots_path = r"./data/tanager/lake_shots/"
tanager_lake_files = read.getFiles(lake_shots_path)

cubeDictLake = {}
for file in tanager_lake_files:
    cubeDictLake[file] = read.getHyperspectralCube('tanager',lake_shots_path,fr"{file}")

# %% slice to forest data for lake shots
slice_info_lake = {
    "20250903_164714_08_4001_ortho_sr_hdf5.h5": [520,570,400,460],
    "20250919_170233_04_4001_ortho_sr_hdf5.h5": [530,580,370,430],
    "20251001_165049_07_4001_ortho_sr_hdf5.h5": [530,580,350,410],
    "20251015_165255_92_4001_ortho_sr_hdf5.h5": [610,660,540,600],
    "20251029_165455_83_4001_ortho_sr_hdf5.h5": [600,650,540,600],
}

labels = ['Week 0','Week 2', 'Week 4', 'Week 6', 'Week 8']
slice_cubes_lake = {}
for image in [*cubeDictLake.keys()]:
    cubeData = cubeDictLake[image]
    slice_data = slice_info_lake[image]
    slicedCubeData = read.sliceCube(cubeData,slice_data)
    read.plotRGB(slicedCubeData)
    slice_cubes_lake[image] = slicedCubeData

#%% analyze spectra of cubes?

labels = ['Week 0','Week 2', 'Week 4', 'Week 6', 'Week 8']

plt.figure
i = 0
for time in [*slice_cubes_lake.keys()]:
    cube = slice_cubes_lake[time]['cube']
    flat_cube = np.concatenate(cube,axis=0)
    print(np.shape(flat_cube))

    avg_spectra = np.mean(flat_cube,axis=0)
    std_specta = np.std(flat_cube,axis=0)
    wavelength = slice_cubes_lake[time]['wavelengths']

    plt.plot(wavelength,avg_spectra,label=labels[i])
    i += 1

plt.xlabel('Wavelength [nm]')
plt.ylabel('Reflectance')
plt.legend()

i = 0
for time in [*slice_cubes_lake.keys()]:
    if i % 2 == 0:
        cube = slice_cubes_lake[time]['cube']
        flat_cube = np.concatenate(cube,axis=0)
        print(np.shape(flat_cube))

        avg_spectra = np.mean(flat_cube,axis=0)
        std_specta = np.std(flat_cube,axis=0)
        wavelength = slice_cubes_lake[time]['wavelengths']

        plt.figure()
        plt.plot(wavelength,avg_spectra,label=labels[i])
        plt.fill_between(wavelength,avg_spectra-std_specta,avg_spectra+std_specta,alpha=0.2)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Reflectance')
        plt.title(time)
    i += 1

#%%
def getNDVI(cubeData,mode='image'):
    cube = cubeData['cube']
    wavelength = cubeData['wavelengths']
    index860 = ((wavelength-860) > 0).argmax()
    index660 = ((wavelength-660) > 0).argmax()

    ndvi = (cube[:,:,index860]-cube[:,:,index660])/(cube[:,:,index860]+cube[:,:,index660])
    
    if mode == 'image':
        return ndvi
    elif mode == 'list':
        return np.concatenate(ndvi)

labels = ['Week 0','Week 2', 'Week 4', 'Week 6', 'Week 8']

avg_ndvi_list = []
std_ndvi_list = []
max_ndvi_list = []
min_ndvi_list = []

i = 0
ndvi_dict = {}
for time in [*slice_cubes_lake.keys()]:
    cubeData = slice_cubes_lake[time]
    ndvi = getNDVI(cubeData,mode='list')

    ndvi_dict[labels[i]] = ndvi

    avg_ndvi = np.mean(ndvi)
    std_ndvi = np.std(ndvi)
    max_ndvi = np.max(ndvi)
    min_ndvi = np.min(ndvi)

    i += 1


fig, ax = plt.subplots()
ax.boxplot(ndvi_dict.values())
ax.set_xticklabels(ndvi_dict.keys())
ax.set_ylabel('NDVI')





#%% Automated read-in of all files for city shots
city_shots_path = r"./data/tanager/city_shots/"
tanager_city_files = read.getFiles(city_shots_path)

cubeDictCity = {}
for file in tanager_city_files:
    cubeDictCity[file] = read.getHyperspectralCube('tanager',city_shots_path,fr"{file}")

# %% slice to city data for city shots
slice_info_city = {
    "20250704_165208_78_4001_ortho_sr_hdf5.h5": [384,404,283,313],
    "20250801_165548_86_4001_ortho_sr_hdf5.h5": None,
    "20250903_164719_92_4001_ortho_sr_hdf5.h5": [28,48,283,313],
    "20250919_170238_88_4001_ortho_sr_hdf5.h5": [34,54,261,291],
    "20251001_165054_91_4001_ortho_sr_hdf5.h5": [27,47,262,292],
    "20251015_165301_89_4001_ortho_sr_hdf5.h5": [92,112,459,489],
    "20251029_165501_94_4001_ortho_sr_hdf5.h5": [74,94,455,485],

}

slice_cubes_city = {}
for image in [*cubeDictCity.keys()]:
    cubeData = cubeDictCity[image]
    slice_data = slice_info_city[image]
    if slice_data != None:
        slicedCubeData = read.sliceCube(cubeData,slice_data)
        read.plotRGB(slicedCubeData)
        slice_cubes_city[image] = slicedCubeData

# %% Generate eigenvalues for each image in each embedding type and read out to files for vegetation
laplacian_eigendata_veg = pd.DataFrame(columns=['Date','Eigenvalues'])
isomap_eigendata_veg = pd.DataFrame(columns=['Date','Eigenvalues'])
lle_eigendata_veg = pd.DataFrame(columns=['Date','Eigenvalues'])
pca_eigendata_veg = pd.DataFrame(columns=['Date','Eigenvalues'])

start = datetime.now()
print(start)

for veg_image in [*slice_cubes_lake.keys()]:
    image = slice_cubes_lake[veg_image]
    if image != None:
        lap = eigencomp.get_laplacian_evs(image,100,100)
        iso = eigencomp.get_iso_evs(image,100,100)
        lle = eigencomp.get_lle_evs(image,100,100)
        pca = eigencomp.get_pca_evs(image,100)

        lap_ev_df = pd.DataFrame({'Date': veg_image, 'Eigenvalues': lap})
        iso_ev_df = pd.DataFrame({'Date': veg_image, 'Eigenvalues': iso})
        lle_ev_df = pd.DataFrame({'Date': veg_image, 'Eigenvalues': lle})
        pca_ev_df = pd.DataFrame({'Date': veg_image, 'Eigenvalues': pca})

        laplacian_eigendata_veg = pd.concat([laplacian_eigendata_veg, lap_ev_df], ignore_index=True)
        isomap_eigendata_veg = pd.concat([isomap_eigendata_veg, iso_ev_df], ignore_index=True)
        lle_eigendata_veg = pd.concat([lle_eigendata_veg, lle_ev_df], ignore_index=True)
        pca_eigendata_veg = pd.concat([pca_eigendata_veg, pca_ev_df], ignore_index=True)

laplacian_eigendata_veg.to_csv('veg_laplacian_eigenvalues.csv', index=False)
isomap_eigendata_veg.to_csv('veg_isomap_eigenvalues.csv', index=False)
lle_eigendata_veg.to_csv('veg_lle_eigenvalues.csv', index=False)
pca_eigendata_veg.to_csv('veg_pca_eigenvalues.csv', index=False)

end = datetime.now()
print(end)
print(f"Total elapsed time: {end-start}")

# %% Generate eigenvalues for each image in each embedding type and read out to files for city
laplacian_eigendata_city = pd.DataFrame(columns=['Date','Eigenvalues'])
isomap_eigendata_city = pd.DataFrame(columns=['Date','Eigenvalues'])
lle_eigendata_city = pd.DataFrame(columns=['Date','Eigenvalues'])
pca_eigendata_city = pd.DataFrame(columns=['Date','Eigenvalues'])

start = datetime.now()
print(start)

for city_image in [*slice_cubes_city.keys()]:
    image = slice_cubes_city[city_image]
    if image != None:
        lap = eigencomp.get_laplacian_evs(image,100,100)
        iso = eigencomp.get_iso_evs(image,100,100)
        lle = eigencomp.get_lle_evs(image,100,100)
        pca = eigencomp.get_pca_evs(image,100)

        lap_ev_df = pd.DataFrame({'Date': city_image, 'Eigenvalues': lap})
        iso_ev_df = pd.DataFrame({'Date': city_image, 'Eigenvalues': iso})
        lle_ev_df = pd.DataFrame({'Date': city_image, 'Eigenvalues': lle})
        pca_ev_df = pd.DataFrame({'Date': city_image, 'Eigenvalues': pca})

        laplacian_eigendata_city = pd.concat([laplacian_eigendata_city, lap_ev_df], ignore_index=True)
        isomap_eigendata_city = pd.concat([isomap_eigendata_city, iso_ev_df], ignore_index=True)
        lle_eigendata_city = pd.concat([lle_eigendata_city, lle_ev_df], ignore_index=True)
        pca_eigendata_city = pd.concat([pca_eigendata_city, pca_ev_df], ignore_index=True)

laplacian_eigendata_city.to_csv('city_laplacian_eigenvalues.csv', index=False)
isomap_eigendata_city.to_csv('city_isomap_eigenvalues.csv', index=False)
lle_eigendata_city.to_csv('city_lle_eigenvalues.csv', index=False)
pca_eigendata_city.to_csv('city_pca_eigenvalues.csv', index=False)

end = datetime.now()
print(end)
print(f"Total elapsed time: {end-start}")
# %%
