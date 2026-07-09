#%% utils
# package imports
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import h5py
import random
# custom code imports
import utils.read as read
import utils.eigencomp as eigencomp

#%%

disease_shots_path = r"/dirs/data/darpa-bto/beets/2022/2022_beets_disease_nano.h5"

healthy_shots_path = r"/dirs/data/darpa-bto/beets/2022/2022_beets_healthy_nano.h5"

#%%
def print_structure(name, obj):
    # Shift text over based on depth to create a visual tree
    indent = "  " * name.count('/')
    
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 Dataset: {name} (shape={obj.shape}, dtype={obj.dtype})")

# Load your file in read-only mode
with h5py.File(healthy_shots_path) as f:
    print("📁 Root File Structure:")
    f.visititems(print_structure)

#%% read-in beets data
def readBeets(file_path,mode='unhealthy'):
    beetsData = {}
    if mode == 'unhealthy':
        with h5py.File(file_path) as f:
            beetsData['wavelengths'] = np.linspace(400,1000,272)
            for plot in [*f['plots'].keys()]:
                plotData = {}
                for day in [*f['plots'][plot].keys()]:
                    spectrum = f['plots'][plot][day]['spectra'][()]
                    disease_severity = f['plots'][plot][day]['ground_truth_severity'][()]
                    cube = f['plots'][plot][day]['cube'][()]
                    mask = f['plots'][plot][day]['mask'][()]
                    plotData[day] = {
                        'spectrum': spectrum,
                        'disease_severity': disease_severity,
                        'cube': cube,
                        'mask': mask
                    }
                beetsData[plot] = plotData
        return beetsData
    elif mode == 'healthy':
        with h5py.File(file_path) as f:
            beetsData['wavelengths'] = np.linspace(400,1000,272)

            for day in [*f['healthy'].keys()]:
                spectrum = f['healthy'][day]['spectra'][()]
                beetsData[day] = {
                    'spectrum': spectrum,
                }
        return beetsData

def getNDVI(wavelengths,spectrum):
    index860 = ((wavelengths-860) > 0).argmax()
    index660 = ((wavelengths-660) > 0).argmax()

    ndvi = (spectrum[:,index860]-spectrum[:,index660])/(spectrum[:,index860]+spectrum[:,index660])
    
    return ndvi

def downsample_2d_axis(arr, axis=0, factor=4):
    # generated using gemini
    if axis == 1:
        # Downsample columns
        valid_cols = (arr.shape[1] // factor) * factor
        cropped = arr[:, :valid_cols]
        reshaped = cropped.reshape(cropped.shape[0], cropped.shape[1] // factor, factor)
        return reshaped.mean(axis=2)
    elif axis == 0:
        # Downsample rows
        valid_rows = (arr.shape[0] // factor) * factor
        cropped = arr[:valid_rows, :]
        reshaped = cropped.reshape(cropped.shape[0] // factor, factor, cropped.shape[1])
        return reshaped.mean(axis=1)
    else:
        raise ValueError("Axis must be 0 or 1 for a 2D array.")

#%%
beets_test = readBeets(disease_shots_path)
beets_healthy = readBeets(healthy_shots_path,mode='healthy')

#%% analyze spectra of plots

plots = [[*beets_test.keys()][10]]

for plot in plots:
    if plot != 'wavelengths':
        plot_data = beets_test[plot]
        plt.figure()
        for day in [*plot_data.keys()]:
            wavelengths = beets_test['wavelengths']
            spectrum = plot_data[day]['spectrum']
            ndvi = getNDVI(wavelengths,spectrum)
            plt.plot(wavelengths,np.mean(spectrum,axis=0),
                    label=f"{day}, disease severity = {plot_data[day]['disease_severity']:.2f}, mean NDVI = {np.mean(ndvi):.2f}")
        plt.legend()
        plt.xlabel('Wavelength')
        plt.ylabel('Mean Reflectance Spectrum')
        plt.title(plot)

#%% fix this later but not currently operational

avg_ndvi_list = []
std_ndvi_list = []
max_ndvi_list = []
min_ndvi_list = []

i = 0
ndvi_dict = {}
plot = [[*beets_test.keys()][5]]
for plot in plots:
    if plot != 'wavelengths':
        plot_data = beets_test[plot]
        plt.figure()
        for day in [*plot_data.keys()]:
            wavelengths = beets_test['wavelengths']
            spectrum = plot_data[day]['spectrum']
            ndvi = getNDVI(wavelengths,spectrum)

            ndvi_dict[day] = ndvi

    avg_ndvi = np.mean(ndvi)
    std_ndvi = np.std(ndvi)
    max_ndvi = np.max(ndvi)
    min_ndvi = np.min(ndvi)

    i += 1


fig, ax = plt.subplots()
ax.boxplot(ndvi_dict.values())
ax.set_xticklabels(ndvi_dict.keys())
ax.set_ylabel('NDVI')

# %% Generate eigenvalues for each image in each embedding type and read out to files for vegetation
plots = [*beets_test.keys()] # plot 5 is pretty good

for plot in plots:
    print(f"Starting {plot}")
    save_array = np.empty((5, 271))
    if plot != 'wavelengths':
        plot_data = beets_test[plot]
        i = 0
        for day in [*plot_data.keys()]:
            eigen_list = []
            eigen_list_d1 = []
            eigen_list_d2 = []
            wavelengths = beets_test['wavelengths']
            spectrum = plot_data[day]['spectrum']
            ndvi = getNDVI(wavelengths,spectrum)
            iso = eigencomp.get_iso_evs(spectrum,10,271)
            eigen_list += [iso]
            save_array[i,:] = np.array(eigen_list)
            i += 1
            #plt.plot(iso/np.linalg.norm(iso),
            #        label=f"{day}, disease severity = {plot_data[day]['disease_severity']:.2f}, mean NDVI = {np.mean(ndvi):.2f}")
        plt.legend()
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue Magnitude')
        plt.yscale('log')
        print(f'Saving {plot}')
        np.save(f'disease_plot_isos_{plot}.npy', save_array)
#%% save disease severity datta

plots = [*beets_test.keys()] # plot 5 is pretty good

for plot in plots:
    print(f"Starting {plot}")
    save_array = np.empty((5))
    if plot != 'wavelengths':
        plot_data = beets_test[plot]
        i = 0
        for day in [*plot_data.keys()]:
            disease_severity = plot_data[day]['disease_severity']
            save_array[i] =disease_severity
            i += 1
        print(f'Saving severity for {plot}...')
        np.save(f'disease_severity_{plot}.npy', save_array)

#%%
diff_list = eigencomp.get_metrics_from_list(eigen_list,'euclidean',mode='diff')
start_list = eigencomp.get_metrics_from_list(eigen_list,'euclidean',mode='start')
plt.figure()
plt.plot(diff_list,label='diff')
plt.plot(start_list,label='start')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Difference Metric')

#%% Downsample spectrum test

plots = [[*beets_test.keys()][1]]

for plot in plots:
    if plot != 'wavelengths':
        plot_data = beets_test[plot]
        for day in [*plot_data.keys()]:
            plt.figure()
            spectrum = plot_data[day]['spectrum']
            disease_severity = plot_data[day]['disease_severity']
            plt.plot(np.mean(spectrum,axis=0),label='measured')
            downsample = downsample_2d_axis(spectrum)
            plt.plot(np.mean(downsample,axis=0),label='downsample 1')
            downsample2 = downsample_2d_axis(downsample)
            plt.plot(np.mean(downsample2,axis=0),label='downsample 2')
            plt.legend()

#%% Downsample eigenvalue test - ISOMAP

plots = [[*beets_test.keys()][1]]

for plot in plots:
    if plot != 'wavelengths':
        plot_data = beets_test[plot]
        for day in [*plot_data.keys()]:
            plt.figure()
            spectrum = plot_data[day]['spectrum']
            iso_spectrum = eigencomp.get_iso_evs(spectrum,10,271)
            disease_severity = plot_data[day]['disease_severity']
            downsample = downsample_2d_axis(spectrum)
            iso_downsample1 = eigencomp.get_iso_evs(downsample,10,271)
            downsample2 = downsample_2d_axis(downsample)
            iso_downsample2 = eigencomp.get_iso_evs(downsample2,10,271)
            
            plt.plot(iso_spectrum/np.linalg.norm(iso_spectrum),label='measured')
            plt.plot(iso_downsample1/np.linalg.norm(iso_downsample1),label='downsample by 2')
            plt.plot(iso_downsample2/np.linalg.norm(iso_downsample2),label='downsample by 4')

            plt.title(f"day = {day}, disease severity = {plot_data[day]['disease_severity']}")
            plt.yscale('log')
            plt.legend()
#%% Downsample eigenvalue test - PCA
for plot in plots:
    if plot != 'wavelengths':
        plot_data = beets_test[plot]
        for day in [*plot_data.keys()]:
            plt.figure()
            spectrum = plot_data[day]['spectrum']
            pca_spectrum = eigencomp.get_pca_evs(spectrum,271)
            disease_severity = plot_data[day]['disease_severity']
            downsample = downsample_2d_axis(spectrum)
            pca_downsample1 = eigencomp.get_pca_evs(downsample,271)
            downsample2 = downsample_2d_axis(downsample)
            pca_downsample2 = eigencomp.get_pca_evs(downsample2,271)
            
            plt.plot(pca_spectrum/np.linalg.norm(pca_spectrum),label='measured')
            plt.plot(pca_downsample1/np.linalg.norm(pca_downsample1),label='downsample by 2')
            plt.plot(pca_downsample2/np.linalg.norm(pca_downsample2),label='downsample by 4')

            plt.title(f"day = {day}, disease severity = {plot_data[day]['disease_severity']}")
            plt.yscale('log')
            plt.legend()

# %% Plot healthy data to verify health status

for i in range(5):

    rng = np.random.default_rng()
    indices = np.random.choice(beets_healthy['D5']['spectrum'].shape[0], size=10000, replace=False)
    random_healthy = beets_healthy['D5']['spectrum'][indices,:]
    healthy_1 = random_healthy[0:5000]
    healthy_2 = random_healthy[5000::]
    plt.figure()
    plt.plot(np.mean(healthy_1,axis=0))
    plt.plot(np.mean(healthy_2,axis=0))

# %% Plot healthy data eigenvalues

for i in range(5):

    rng = np.random.default_rng()
    indices = np.random.choice(beets_healthy['D5']['spectrum'].shape[0], size=10000, replace=False)
    random_healthy = beets_healthy['D5']['spectrum'][indices,:]
    healthy_1 = random_healthy[0:5000]
    healthy_2 = random_healthy[5000::]

    healthy_1_isos = eigencomp.get_iso_evs(healthy_1,10,271)
    healthy_2_isos = eigencomp.get_iso_evs(healthy_2,10,271)

    plt.figure()
    plt.plot(healthy_1_isos/np.linalg.norm(healthy_1_isos),label='healthy 1')
    plt.plot(healthy_2_isos/np.linalg.norm(healthy_2_isos),label = 'healthy 2')
    plt.yscale('log')
# %% Batch run healthy eigenvalues stats

batch_healthy_list = []
for i in range(100):
    print(f"Beginning run {i+1}")
    rng = np.random.default_rng()
    indices = np.random.choice(beets_healthy['D5']['spectrum'].shape[0], size=10000, replace=False)
    random_healthy = beets_healthy['D5']['spectrum'][indices,:]
    healthy_1 = random_healthy[0:5000]
    healthy_2 = random_healthy[5000::]

    healthy_1_isos = eigencomp.get_iso_evs(healthy_1,10,271)
    healthy_1_isos = healthy_1_isos/np.linalg.norm(healthy_1_isos)
    healthy_2_isos = eigencomp.get_iso_evs(healthy_2,10,271)
    healthy_2_isos = healthy_2_isos/np.linalg.norm(healthy_2_isos)

    # manually calculate manhattan distance
    metric = np.sum(abs(healthy_1_isos-healthy_2_isos))
np.save(f'healthy_plot_isos_day_D5.npy', np.array(batch_healthy_list))

# %% NOT STARTED YET: plot false color images

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

allCubeDataVeg = []
for image in [*slice_cubes_lake.keys()]:
    cube = slice_cubes_lake[image]['cube']
    flat_cube = np.concatenate(cube,axis=0)
    #flat_cube_partial = flat_cube[0:int(len(flat_cube)/2)]
    flat_cube_partial = flat_cube
    allCubeDataVeg += [flat_cube_partial]

allCubeDataVeg = np.array(allCubeDataVeg)
allCubeDataVeg = allCubeDataVeg[2::]
allCubeDataVeg = np.concatenate(allCubeDataVeg,axis=0)
print(np.shape(allCubeDataVeg))

#%% reshape test

reg_day1 = allCubeDataVeg[0:3000]
reg_day2 = allCubeDataVeg[3000:6000]
reg_day3 = allCubeDataVeg[6000::]

reg_day1_reshape = np.reshape(reg_day1,(50,60,421),order='C')
reg_day2_reshape = np.reshape(reg_day2,(50,60,421),order='C')
reg_day3_reshape = np.reshape(reg_day3,(50,60,421),order='C')

def plotRGBReshape(cube,wavelengths):

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

    colorImage = colorImage.swapaxes(0,2)

    fig = plt.figure()
    plt.imshow(colorImage)
    plt.show()

plotRGBReshape(reg_day1_reshape,slice_cubes_lake[image]['wavelengths'])

#%%
iso = Isomap(n_neighbors=300,n_components=100,metric='euclidean')
iso_fit = iso.fit_transform(allCubeDataVeg)

#%%
pca = PCA(n_components=100)
pca_fit = pca.fit_transform(allCubeDataVeg)
#%%
fit_day1 = iso_fit[0:3000]
fit_day2 = iso_fit[3000:6000]
fit_day3 = iso_fit[6000::]

pfit_day1 = pca_fit[0:3000]
pfit_day2 = pca_fit[3000:6000]
pfit_day3 = pca_fit[6000::]

#%% Plot Isomap and PCA slices

start = 3
plt.figure(1)
plt.scatter(fit_day1[:,start],fit_day1[:,start+1],label='Week 0')
plt.scatter(fit_day2[:,start],fit_day2[:,start+1],label = 'Week 2')
plt.scatter(fit_day3[:,start],fit_day3[:,start+1],label = 'Week 4')
plt.legend()
#plt.title('Isomap of Data from Both Days Fit To 08/10 Isomap')
plt.xlabel(f'Latent Dimension {start} (X)')
plt.ylabel(f'Latent Dimension {start+1} (Y)')
plt.title('ISOMAP')
plt.show()

plt.figure(2)
plt.scatter(pfit_day1[:,start],pfit_day1[:,start+1],label='Week 0')
plt.scatter(pfit_day2[:,start],pfit_day2[:,start+1],label = 'Week 2')
plt.scatter(pfit_day3[:,start],pfit_day3[:,start+1],label = 'Week 4')
plt.legend()
#plt.title('Isomap of Data from Both Days Fit To 08/10 Isomap')
plt.xlabel(f'Principal Component {start} (X)')
plt.ylabel(f'Principal Component {start+1} (Y)')
plt.title('PCA')
plt.show()



# 0, 1,2,4,5?

# dimensions 2 and 3 are pretty separable
# need to figure out how to reshape array into workable image!
# I think with numpy reshape this will work if you use the spectra as the first value? Test on regular flat array first to see if it looks right
# %%

fit_day1_reshape = np.reshape(fit_day1,(50,60,100),order='C')
fit_day2_reshape = np.reshape(fit_day2,(50,60,100),order='C')
fit_day3_reshape = np.reshape(fit_day3,(50,60,100),order='C')

pfit_day1_reshape = np.reshape(pfit_day1,(50,60,100),order='C')
pfit_day2_reshape = np.reshape(pfit_day2,(50,60,100),order='C')
pfit_day3_reshape = np.reshape(pfit_day3,(50,60,100),order='C')

def plotFalseColorEmbedding(emb_cube,dims=(0,1,2)):
    colorData = np.array([emb_cube[:,:,dims[0]],emb_cube[:,:,dims[1]],emb_cube[:,:,dims[2]]])
    # normalize
    colorMin = np.min(np.concatenate(np.concatenate(colorData)))
    colorData = colorData + abs(colorMin)
    colorMax = np.max(np.concatenate(np.concatenate(colorData)))
    colorImage = colorData/colorMax
    colorImage = np.swapaxes(colorImage,0,2)
    #colorImage = np.swapaxes(colorImage,0,1)
    fig = plt.figure()
    plt.imshow(colorImage)
    plt.show()

#%%

dims = (2,4,5)

print('ISOMAP')
for fit in [fit_day1_reshape,fit_day2_reshape,fit_day3_reshape]:
    plotFalseColorEmbedding(fit,dims=dims)
print('PCA')
for fit in [pfit_day1_reshape,pfit_day2_reshape,pfit_day3_reshape]:
    plotFalseColorEmbedding(fit,dims=dims)

#%% fcir normalization routine?
wvl=cubeData['wavelengths']
frIndex = ((wvl-860) > 0).argmax()
fgIndex = ((wvl-660) > 0).argmax()
fbIndex = ((wvl-560) > 0).argmax()

imgs = [reg_day1_reshape,
    reg_day2_reshape,
    reg_day3_reshape]

fcirImgs = []
for img in imgs:
    colorData = np.array([img[:,:,frIndex],img[:,:,fgIndex],img[:,:,fbIndex]])
    fcirImgs += [colorData]
fcirImgs = np.array(fcirImgs)

fcirRed = fcirImgs[:,0,:,:]
fcirGreen = fcirImgs[:,1,:,:]
fcirBlue = fcirImgs[:,2,:,:]

fcirRedMax = np.max(np.reshape(fcirRed,-1))
fcirGreenMax = np.max(np.reshape(fcirGreen,-1))
fcirBlueMax = np.max(np.reshape(fcirBlue,-1))

fcirImgsNorm = np.array([fcirRed/fcirRedMax,fcirGreen/fcirGreenMax,fcirBlue/fcirBlueMax])
print(np.shape(fcirImgsNorm))
fcirImgsNorm = np.swapaxes(fcirImgsNorm,0,1)
fcirImgsNorm = np.swapaxes(fcirImgsNorm,2,3)
# %% make my big jumbo plot for my paper

rgb_fits = [
    reg_day1_reshape,
    reg_day2_reshape,
    reg_day3_reshape,
    fcirImgsNorm[0],
    fcirImgsNorm[1],
    fcirImgsNorm[2],
    pfit_day1_reshape,
    pfit_day2_reshape,
    pfit_day3_reshape,
    fit_day1_reshape,
    fit_day2_reshape,
    fit_day3_reshape,
]

wvl=cubeData['wavelengths']
rIndex = ((wvl-650) > 0).argmax()
gIndex = ((wvl-550) > 0).argmax()
bIndex = ((wvl-450) > 0).argmax()

frIndex = ((wvl-860) > 0).argmax()
fgIndex = ((wvl-660) > 0).argmax()
fbIndex = ((wvl-560) > 0).argmax()

dims = (2,4,5)

perBand = True

rgb_images = []
i = 0
for img in rgb_fits:
    if np.shape(img)[0] == 3:
        img = np.swapaxes(img,1,2)
    if np.shape(img)[2] == 100:
        colorData = np.array([img[:,:,dims[0]],img[:,:,dims[1]],img[:,:,dims[2]]])
    else:
        if i < 3:
            colorData = np.array([img[:,:,rIndex],img[:,:,gIndex],img[:,:,bIndex]])
            perBand = False
        else:
            colorData = img
            perBand = False
    # normalize
    colorMin = np.min(np.concatenate(np.concatenate(colorData)))
    if colorMin < 0:
        colorData = colorData + abs(colorMin)
    if perBand:
        colorDataNorm = []
        for color in colorData:
            colorNorm = color/np.max(np.concatenate(color))
            colorDataNorm += [colorNorm]
        colorDataNorm = np.array(colorDataNorm)
    else:
        colorMax = np.max(np.concatenate(np.concatenate(colorData)))
        # fix normalization procedure for color max to do it per band?
        colorDataNorm = colorData/colorMax
    colorImage = np.swapaxes(colorDataNorm,0,2)
    print(np.shape(colorImage))
    rgb_images += [colorImage]
    i += 1

#%%

in1 = 3
in2 = 3

fig, axs = plt.subplots(3, 4)
i = 0
panel_font = {'family': 'serif', 'size': 12,'name': 'Times New Roman'}

labels = ['Week 0','Week 2','Week 4']



for rgb_img in rgb_images:
    plt.tick_params(labelbottom=False, labelleft=False)
    axs[i % in2, int(i/in1) % 4].imshow(np.swapaxes(rgb_img,0,1))
    axs[i % in2, int(i/in1) % 4].set_xticklabels([])
    axs[i % in2, int(i/in1) % 4].set_yticklabels([])
    axs[i % in2, int(i/in1) % 4].set_xticks([])
    axs[i % in2, int(i/in1) % 4].set_yticks([])
    axs[i % in2, int(i/in1) % 4].text(0.65, -0.25, f'({chr(97+i)})', transform=axs[int(i/4) % 3, i % 4].transAxes, 
            fontdict=panel_font, va='bottom', ha='right')

    if i < 3:
        axs[i % in2, int(i/in1) % 4].set_ylabel(labels[i],fontname='Times New Roman',fontsize=12)
    i += 1

#plt.tight_layout()
fig.subplots_adjust(wspace=0.1, hspace=0.05)
fig.show()
fig.savefig('false_color.png', dpi=300, bbox_inches='tight') 
# %%
