#%% imports
import numpy as np
import spectral.io.envi as envi
import plotly.express as px
import matplotlib.pyplot as plt
import h5py
import os

#%% read-in functions

def getFiles(file_path):
    files = []
    for entry in os.listdir(file_path):
        full_path = os.path.join(file_path, entry)
        if os.path.isfile(full_path):
            files.append(entry)
    return files

def getHyperspectralCube(sensor,file_path,file_name,header_name=None):
    # reads in hyperspectral cube with associated metadata

    file = file_path+file_name

    if sensor == 'spectir':
        if header_name is not None:
            img = envi.open(file,file_path+header_name)
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
        
    elif sensor == 'tanager':
            with h5py.File(file, 'r') as f:
                dataset = f[[*f.keys()][0]]['GRIDS']['HYP']['Data Fields']['surface_reflectance']
                reflectance = dataset[()]
                mask = (reflectance < 0)
                reflectance[mask] = 0
                reflectance = np.swapaxes(reflectance,0,1)
                
                wavelengths = dataset.attrs['wavelengths']
                good_wavelengths = dataset.attrs['good_wavelengths']

                reflectance=reflectance*good_wavelengths[np.newaxis,:,np.newaxis]

                cirrus = f[[*f.keys()][0]]['GRIDS']['HYP']['Data Fields']['beta_cirrus_mask'][()]
                cirrus_mask = np.ones(np.shape(cirrus))-cirrus
                cloud = f[[*f.keys()][0]]['GRIDS']['HYP']['Data Fields']['beta_cloud_mask'][()]
                cloud_mask = np.ones(np.shape(cloud))-cloud
                aerosol = f[[*f.keys()][0]]['GRIDS']['HYP']['Data Fields']['aerosol_optical_depth'][()]
                water_vapor = f[[*f.keys()][0]]['GRIDS']['HYP']['Data Fields']['column_water_vapour'][()]

                reflectance = np.swapaxes(reflectance,1,2)
                reflectance_masked = reflectance*cloud_mask[:,:,np.newaxis]*cirrus_mask[:,:,np.newaxis],
                return {
            'cube': reflectance_masked[0][:,:,:-5],
            'wavelengths': wavelengths[:-5],
            'file_name': file_name,
            'cirrus_mask': cirrus_mask,
            'cloud_mask': cloud_mask,
            'aerosol_depth': aerosol,
            'water_vapor': water_vapor
            }

def sliceCube(cubeData,slice):
    # slice should be a 4-element list with the coordinates slicing the array
    newCubeData = cubeData.copy()
    hyperCube = newCubeData['cube']
    aerosol = newCubeData['aerosol_depth']
    water = newCubeData['water_vapor']
    slicedHyperCube = hyperCube[slice[0]:slice[1],slice[2]:slice[3],:]
    slicedAerosol = aerosol[slice[0]:slice[1],slice[2]:slice[3]]
    slicedWater = water[slice[0]:slice[1],slice[2]:slice[3]]
    newCubeData['cube'] = slicedHyperCube
    newCubeData['aerosol_depth'] = slicedAerosol
    newCubeData['water_vapor'] = slicedWater
    return newCubeData

# plotting functions
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

# misc 
def downsample_avg(arr, factor):
    # generated using Goolge Gemini
    sh = arr.shape
    # Reshape to (new_ax0, factor, new_ax1, factor, depth)
    reshaped = arr.reshape(sh[0]//factor, factor, sh[1]//factor, factor, sh[2])
    # Mean across the factor axes (1 and 3)
    return reshaped.mean(axis=(1, 3))