# -*- coding: utf-8 -*-
"""
Tomographic reconstruction utilities for CHESS data and interactive parameter tuning.
"""
import os

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import tomopy
from matplotlib import animation
from matplotlib.widgets import Button, Slider, TextBox
from skimage import transform as sktransform
from skimage import util as skutil


def loadHDF5(fileName, sinogramName='tomoImgs', thetaName='theta'):
    """
    Loads tomographic data and theta from an hdf5 file.
    Tomo data should be in the format [elements, rows, angles, cols]
    Theta should be arr of radians

    Parameters
    ----------
    filename : string
        file to parse
    sinogramName : string
        name of tomographic dataset
    thetaName : string
        name of theta dataset

    Returns
    -------
    array
        [tomographic data, theta]
    """
    with h5py.File(fileName, 'r') as f:
        tomoImgs = f[sinogramName][:]
        theta = f[thetaName][:]
    return tomoImgs, theta


def _guessVBounds(image, area=1):
    """
    Estimates lower and upper bounds of color spectrum for given array

    Parameters
    ----------
    image : 2D array_like
        image to plot
    area : scalar (0,1]
        % of pixels to maintain within generated bounds

    Returns
    -------
    array
        [estimated lower bound, estimated upper bound]
    """
    if len(image) == 1:
        return _guessVBounds(image[0], area)

    image = image[len(image)//4:len(image)*3//4,
                  len(image[0])//4:len(image[0])*3//4]
    flatten = np.nan_to_num(image.flatten())
    valid = flatten[np.isfinite(flatten)]
    if valid.size == 0:
        return 0, 1

    n, bins = np.histogram(valid, 100, range=(valid.min(), valid.max()))
    total = float(n.sum())
    if total == 0:
        return 0, 1

    mask = n / total > 0.005
    bins = bins[:-1][mask]
    n = n[mask]
    if len(bins) == 0:
        return 0, 1

    counted = 0.0
    init_min = 0
    init_max = 0
    for x in range(len(bins)):
        if x % 2 == 0 and counted / total < 1 - area:
            counted += n[x // 2]
            init_min += 1
        elif x % 2 == 1 and counted / total < 1 - area:
            counted += n[len(bins) - x // 2 - 1]
            init_max += 1
        else:
            break

    vmin = bins[init_min]
    vmax = bins[len(bins) - init_max - 1]
    if vmin >= vmax:
        print("Couldn't generate vmin/vmax bounds with given inputs!")
        return 0, 1
    return vmin, vmax


def _toSci(num):
    """
    converts float to scientific notation

    Parameters
    ----------
    num : float
        number to convert

    Returns
    -------
    String
        Scientific notation of num
    """
    if num == 0:
        return '0.0e0'
    if num < 0:
        return '-' + _toSci(-num)
    exponent = int(np.floor(np.log10(num)))
    mantissa = round(10 ** (-exponent) * num, 1)
    return f'{mantissa}e{exponent}'


def _normalize_recon_params(layers, centers):
    """Return layer bounds and per-layer center offsets."""
    if np.isscalar(layers):
        layers = [layers, layers + 1]
    num_layers = layers[1] - layers[0]

    if np.isscalar(centers):
        centers = np.full(num_layers, centers)
    elif len(centers) == 2:
        assert num_layers >= 1, "Too many centers for number of layers!"
        centers = np.linspace(centers[0], centers[1], num=num_layers)
    else:
        assert len(centers) == num_layers, "unequal # of centers and layers!"
    return layers, centers


def _reconstruct_layer(sinogram, theta, center, image_bounds, sigma, ncore, algorithm,
                       run_secondary_sirt, secondary_iter):
    """Reconstruct a single sinogram slice with optional SIRT refinement."""
    width = image_bounds[1] - image_bounds[0]
    tomopy_center = width / 2.0 + center
    cleaned = tomopy.prep.stripe.remove_stripe_fw(
        sinogram, sigma=sigma, ncore=ncore)
    initial = tomopy.recon(
        cleaned, theta, center=tomopy_center, algorithm=algorithm,
        sinogram_order=False, ncore=ncore)

    if not run_secondary_sirt:
        return initial[0]

    options = {'proj_type': 'cuda', 'method': 'SIRT_CUDA', 'num_iter': secondary_iter}
    refined = tomopy.recon(
        cleaned, theta, center=tomopy_center, init_recon=initial,
        algorithm=tomopy.astra, options=options, sinogram_order=False, ncore=ncore)
    return refined[0]


def reconstruct(sinograms, centers, imageBounds, layers, theta, sigma=.1, ncore=4,
                algorithm='gridrec', run_secondary_sirt=False, secondary_iter=100):
    """
    Reconstruct object from projection data.

    Accepts sinograms shaped [elements, rows, angles, cols] or [rows, angles, cols]
    and returns a 4D reconstruction [elements, layers, height, width].
    """
    layers, centers = _normalize_recon_params(layers, centers)
    width = imageBounds[1] - imageBounds[0]
    height = imageBounds[1] - imageBounds[0]
    num_layers = layers[1] - layers[0]
    col_slice = slice(imageBounds[0], imageBounds[1])

    if sinograms.ndim == 3:
        sinograms = np.swapaxes(sinograms, 0, 1)[np.newaxis, ...]
    elif sinograms.ndim == 4:
        sinograms = np.swapaxes(sinograms, 1, 2)
    else:
        raise ValueError(f"Expected 3D or 4D sinograms, got shape {sinograms.shape}")

    recon_clean = np.zeros([len(sinograms), num_layers, height, width])
    for el in range(len(sinograms)):
        if len(sinograms) > 1:
            print(f"Process Element #{el}")
        element_recons = np.zeros([num_layers, height, width])
        for layer_idx in range(num_layers):
            if num_layers > 1:
                print(f"Processing Layer #{layer_idx}")
            layer = layers[0] + layer_idx
            sinogram = sinograms[el, :, layer:layer + 1, col_slice]
            element_recons[layer_idx] = _reconstruct_layer(
                sinogram, theta, centers[layer_idx], imageBounds, sigma, ncore,
                algorithm, run_secondary_sirt, secondary_iter)
        recon_clean[el] = tomopy.misc.corr.remove_ring(element_recons, rwidth=17)

    if num_layers != 1:
        print("complete!")
    return recon_clean


class ValHelper:
    """Interactive GUI for tuning reconstruction parameters."""

    def __init__(self, fig2, sinograms, imageBounds, layer, layerBounds, theta,
                 sigma, ncore, algorithm, vmin, vmax, cmap, interpolation):
        self.fig2 = fig2
        self.sinograms = sinograms
        self.center = 0
        self.imageBounds = imageBounds
        self.layerBounds = layerBounds
        self.layer = layer - self.layerBounds[0]
        self.theta = theta
        self.sigma = sigma
        self.ncore = ncore
        self.algorithm = algorithm
        self.cmap = cmap
        self.interpolation = interpolation
        self.element = 0

        self._reconstruct_current()
        if vmin is not None and vmax is not None:
            self.vMin = vmin
            self.vMax = vmax
        else:
            self.vMin, self.vMax = _guessVBounds(self.recon[0])

        self._show_recon()
        self.prevMin = self.vMin
        self.prevMax = self.vMax

        axcolor = 'lightgoldenrodyellow'
        ax_max = plt.axes([0.08, 0.04, 0.65, 0.025], facecolor=axcolor)
        ax_min = plt.axes([0.08, 0.08, 0.65, 0.025], facecolor=axcolor)
        ax_center = plt.axes([0.08, 0.12, 0.65, 0.025], facecolor=axcolor)
        ax_layer = plt.axes([0.08, 0.16, 0.09, 0.025])
        ax_element = plt.axes([0.28, 0.16, 0.09, 0.025])
        ax_center_button = plt.axes([0.46, 0.16, 0.14, 0.025])

        self.prompt = plt.text(1.1, 0, "Ready.")
        self.minSlide = Slider(
            ax_min, 'vMin', self.vMin, self.vMin + (self.vMax - self.vMin) / 2,
            valinit=self.vMin)
        self.maxSlide = Slider(
            ax_max, 'vMax', self.vMin + (self.vMax - self.vMin) / 2, self.vMax,
            valinit=self.vMax)
        self.centerSlide = Slider(
            ax_center, 'center',
            max(-200, (imageBounds[0] - imageBounds[1]) / 2),
            min(200, (imageBounds[1] - imageBounds[0]) / 2),
            valinit=0.0)
        self.minSlide.valtext.set_text(_toSci(self.minSlide.val))
        self.maxSlide.valtext.set_text(_toSci(self.maxSlide.val))
        self.minSlide.on_changed(self.smallUpdate)
        self.maxSlide.on_changed(self.smallUpdate)
        self.centerSlide.on_changed(self.largeUpdate)

        self.layerButton = TextBox(
            ax_layer, "layer ", color=axcolor, initial=str(self.layer))
        self.layerButton.on_submit(self.layerSubmit)
        self.elementButton = TextBox(
            ax_element, "element ", color=axcolor, initial="0")
        self.elementButton.on_submit(self.elementSubmit)
        self.centerButton = TextBox(
            ax_center_button, "center ", color=axcolor, initial="0")
        self.centerButton.on_submit(self.centerSubmit)

        ax_undo = plt.axes([0.85, 0.05, 0.09, 0.06])
        self.undoButton = Button(
            ax_undo, 'Undo', color=axcolor, hovercolor='0.975')
        self.undoButton.on_clicked(self.undo)
        self.versions = [[self.center, self.imageBounds,
                          self.layer, self.vMin, self.vMax, self.element]]

    def _current_layer(self):
        return self.layer + self.layerBounds[0]

    def _reconstruct_current(self):
        self.recon = reconstruct(
            self.sinograms[self.element], self.center, self.imageBounds,
            self._current_layer(), self.theta, self.sigma, self.ncore, self.algorithm)

    def _show_recon(self, recon=None, vmin=None, vmax=None):
        if recon is None:
            recon = self.recon
        if vmin is None:
            vmin = self.vMin
        if vmax is None:
            vmax = self.vMax
        self.fig2.imshow(
            ndi.gaussian_filter(recon[0][0], 0.75),
            vmin=vmin, vmax=vmax, cmap=self.cmap, interpolation=self.interpolation)

    def _record_version(self):
        self.versions.append(
            [self.center, self.imageBounds, self.layer,
             self.vMin, self.vMax, self.element])

    def largeUpdate(self, val):
        self.center = self.centerSlide.val
        self._reconstruct_current()
        self._show_recon()
        self._record_version()
        self.prompt.set_text("Center Changed.")

    def smallUpdate(self, val):
        self.vMin = self.minSlide.val
        self.vMax = self.maxSlide.val
        self._show_recon(vmin=self.vMin, vmax=self.vMax)
        self.minSlide.valtext.set_text(_toSci(self.vMin))
        self.maxSlide.valtext.set_text(_toSci(self.vMax))
        self.prompt.set_text("vMin/vMax Changed.")
        self._record_version()

    def layerSubmit(self, text):
        try:
            self.layer = int(text)
            if self.layer in range(self.layerBounds[1] - self.layerBounds[0]):
                self._reconstruct_current()
                self._show_recon()
                self._record_version()
                self.prompt.set_text("Layer Changed.")
            else:
                self.prompt.set_text("Invalid Layer!")
        except ValueError:
            self.prompt.set_text("Not an Integer!")

    def elementSubmit(self, text):
        try:
            el = int(text)
            if el in range(len(self.sinograms)):
                self.element = el
                self._reconstruct_current()
                self._show_recon()
                self._record_version()
                self.prompt.set_text("Element Changed.")
            else:
                self.prompt.set_text("Invalid Element!")
        except ValueError:
            self.prompt.set_text("Not an Integer!")

    def centerSubmit(self, text):
        try:
            if text.startswith('neg(') and text.endswith(')'):
                self.center = -float(text[4:-1])
            else:
                self.center = float(text)
            self.centerSlide.set_val(self.center)
        except ValueError:
            self.prompt.set_text("Not a scalar!")

    def undo(self, val):
        if len(self.versions) < 2:
            self.prompt.set_text("Nothing to Undo!")
            return

        old_ver = self.versions[-2]
        self.recon = reconstruct(
            self.sinograms[old_ver[5]], old_ver[0], old_ver[1],
            old_ver[2] + self.layerBounds[0], self.theta, self.sigma,
            self.ncore, self.algorithm)
        self.center = old_ver[0]
        self.imageBounds = old_ver[1]
        self.layer = old_ver[2]
        self.vMin = old_ver[3]
        self.vMax = old_ver[4]
        self.element = old_ver[5]
        self._show_recon(vmin=self.vMin, vmax=self.vMax)
        self.versions = self.versions[:-1]
        self.prompt.set_text("Undo Complete.")

    def getCenter(self):
        return self.center

    def getLayer(self):
        return self._current_layer()


def launchValHelper(sinograms, imageBounds, layer, layerBounds, theta, sigma=.1,
                     ncore=4, algorithm='gridrec', vmin=None, vmax=None,
                     cmap='gray', interpolation='none'):
    """Launch GUI for interactive reconstruction parameter tuning."""
    plt.close()
    fig2 = plt.axes([0.05, 0.25, 0.7, 0.7])
    return ValHelper(fig2, sinograms, imageBounds, layer, layerBounds, theta,
                     sigma, ncore, algorithm, vmin, vmax, cmap, interpolation)


class BoundHelper:
    """
    Class to help user find sample coordinate vals
    """

    def __init__(self, radiograph, vMin, vMax, cmap, interpolation):
        """
        initalizes variables, sets up mouse listening, and plots radiograph
        """
        self.radiograph = radiograph
        if vMin is None or vMax is None:
            vBounds = _guessVBounds(radiograph, 1)
            self.vMin = vBounds[0]
            self.vMax = vBounds[1]
        else:
            self.vMin = vMin
            self.vMax = vMax

        self.interpolation = interpolation
        self.cmap = cmap
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        plt.imshow(self.radiograph, vmin=self.vMin, vmax=self.vMax,
                   cmap=self.cmap, interpolation=self.interpolation)
        self.c1 = self.fig.canvas.mpl_connect('button_press_event', self)
        self.c2 = self.ax.figure.canvas.mpl_connect(
            'button_release_event', self.onrelease)
        # define texts
        self.prompt = plt.text(len(self.radiograph[0]) // 2, -len(
            self.radiograph)//15, "Pick Top Left Coordinate.", fontsize=15, horizontalalignment='center')
        self.display = plt.text(len(self.radiograph[0]) // 2, len(self.radiograph)*1.1,
                                "Coordinate 1: []      Coordinate 2: []", fontsize=10, horizontalalignment='center')

        self.layerBounds = []
        self.imageBounds = []
        self.temp = []
        self.x = 0
        self.y = 0

    def onrelease(self, event):
        """
        Called when mouse is released.
        If too far from the original press, this method does nothing.
        If first valid click, method stores coordinate values.
        If second valid click, method stores coords and converts to imageBounds and layerBounds.
        """
        x, y = event.xdata, event.ydata
        if(x is not None and y is not None):
            if((self.x-x)**2 + (self.y-y)**2 < 50):

                self.temp += [[int(x), int(y)]]
                if(len(self.temp) <= 2):
                    print(self.temp)
                    self.prompt.set_text("Pick Bottom Right Coordinate.")
                    self.display.set_text(
                        "Coordinate 1: " + str(self.temp[0]) + "     " + " Coordinate 2: []")
                    self.ax.plot(x, y, 'ro')

                if(len(self.temp) == 2):
                    self.layerBounds = [self.temp[0][1], self.temp[1][1]]
                    self.imageBounds = [self.temp[0][0], self.temp[1][0]]
                    rect = patches.Rectangle((self.imageBounds[0], self.layerBounds[0]), self.imageBounds[1]-self.imageBounds[0],
                                             self.layerBounds[1]-self.layerBounds[0], edgecolor="red", facecolor='none', linewidth=1)
                    self.ax.add_patch(rect)
                    self.prompt.set_text("Selected " + str(self.layerBounds[1]-self.layerBounds[0]) + " layers, and " + str(
                        self.imageBounds[1]-self.imageBounds[0]) + " columns.")
                    self.display.set_text(
                        "Coordinate 1: " + str(self.temp[0]) + "     " + " Coordinate 2: " + str(self.temp[1]))

            else:
                self.prompt.set_text("Invalid Coordinates!")
            self.fig.canvas.draw()

    def __call__(self, event):
        """
        called when mouse is clicked. Stores x & y mouse data
        """
        self.x = event.xdata
        self.y = event.ydata

    def getImageBounds(self):
        """
        returns imageBounds coords
        """
        return self.imageBounds

    def getLayerBounds(self):
        """
        returns layerBounds coords
        """
        return self.layerBounds


def launchBoundHelper(radiograph, vMin=None, vMax=None, cmap=None, interpolation='none'):
    """
    Launches GUI which allows users to pick section of sample for evaluation.

    Parameters
    ----------
    radiograph : 2D ndarray
        radiograph to display
    vmin/vmax : scalar
        lower/upper bound of color spectrum
    cmap : Colormap (matplotlib.colors.Colormap)
        function which takes in scalars and outputs colors (determines color scheme)
    interpolation : string
        determines algo for interpolation, see matplotlib.pyplot.imshow for list of algos

    Returns
    -------
    instance of BoundHelper
        instance which holds converted coordinate points
    """
    plt.close()
    return BoundHelper(radiograph, vMin, vMax, cmap, interpolation)


def calcCenters(layerBounds, topCenter, bottomCenter):
    """Linearly interpolate rotation centers across layer bounds."""
    return np.linspace(topCenter, bottomCenter, num=layerBounds[1] - layerBounds[0])


def rotate(imgs, theta):
    """Rotate image(s) counterclockwise by theta degrees."""
    return sktransform.rotate(imgs, theta, preserve_range=True)


def crop(imgs, left, right, top, bottom):
    """Crop image(s) to the given pixel boundaries."""
    height, width = imgs.shape[:2]
    return skutil.crop(imgs, ((top, height - bottom), (left, width - right)))


def save(filename, names, data):
    """Save arrays as named datasets in an HDF5 file."""
    try:
        with h5py.File(str(filename), 'w') as hf:
            for name, array in zip(names, data):
                hf.create_dataset(name, data=array)
        print(f"File saved at {filename}")
    except OSError:
        print("Save failure!")


class Giffer:
    def __init__(self, volume, cmap):
        """
        plots and initiates animation
        """
        self.volume = volume
        self.fig = plt.figure()
        self.index = 0
        bounds = _guessVBounds(volume[len(volume)//2])
        self.im = plt.imshow(
            volume[self.index], vmin=bounds[0], vmax=bounds[1], cmap=cmap, animated=True)
        self.ani = animation.FuncAnimation(
            self.fig, self.updatefig, interval=100, blit=True)
        self.ax = plt.gca()
        self.ax.set_title('Image %d' % (self.index))

    def updatefig(self, *args):
        self.index = (self.index + 1) % self.volume.shape[0]
        self.im.set_array(self.volume[self.index])
        self.ax.set_title('Image %d' % self.index)
        return self.im, self.ax


def multiSliceGiffer(volume, cmap=None):
    """Display that cycles through slices of a volume."""
    Giffer(volume, cmap)


def genElTomo(elementDataFolders, numlayers):
    els = []
    for x in range(len(elementDataFolders)):
        els += [plt.imread(elementDataFolders[x])]
    I = els[0]
    thetasteps = len(I[1])
    numx = len(I[:, 1])/numlayers
    x = np.arange(0, numx)

    data = I[:, :, 1]
    data4d = np.zeros([len(els), numlayers, thetasteps, numx])
    for d in range(len(els)):
        I = els[d]
        data = I[:, :, 1]
        for j in range(0, thetasteps):
            for k in range(0, numlayers):
                for i in range(0, numx):
                    m = i+(numx*k)
                    data4d[d, k, j, i] = data[m, j]

    return data4d

#####F2 LOAD FUNCTIONS####


def _chess_scalar_dir(data_folder):
    return data_folder[:len(data_folder) - 3] + 'scalars'


def _load_chess_metadata(data_folder):
    scalar_dir = _chess_scalar_dir(data_folder)
    summary_dir = os.listdir(scalar_dir)[0]
    return np.loadtxt(os.path.join(scalar_dir, summary_dir, 'summary.dat'))


def _count_negative_theta_skip(theta, start=2):
    num2skip = start
    for angle in theta:
        if angle < 0:
            num2skip += 1
        else:
            break
    return num2skip


def _chess_scan_slice(meta_data, num2skip, num_imgs):
    return slice(num2skip - 2, num2skip + num_imgs - 2)


def genDark(tdf_data_folder, tdf_fold_start=None, num2skip=None,
            tdf_num_imgs=None):
    # if params are given
    if(tdf_fold_start is not None and num2skip is not None and tdf_num_imgs is not None):
        tdf_img_start = tdf_fold_start+num2skip
        tdf_img_nums = np.arange(tdf_img_start, tdf_img_start+tdf_num_imgs, 1)

        tdf_stack = np.zeros([len(tdf_img_nums), 2048, 2048])

        print('Loading data for median dark field...')

        for x in range(len(tdf_img_nums)):
            tdf_stack[x, :, :] = plt.imread(
                tdf_data_folder + 'nf_%0.6d.tif' % (tdf_img_nums[x]))
            # image_stack[x,:,:]=np.flipud(tmp_img>threshold)

        # take the median
        tdf = np.median(tdf_stack, axis=0)
        print('complete!')
        return tdf
    else:
        meta_data = _load_chess_metadata(tdf_data_folder)
        theta = meta_data[:, 1]
        first_image = sorted(os.listdir(tdf_data_folder))[0]
        first_image = first_image[3:first_image.index('.tif')]
        tdf_stack = np.zeros([len(theta), 2048, 2048])

        print('Loading data for median dark field...')
        counter = 0
        for x in range(len(theta) - 1):
            if theta[x] > 0:
                tdf_stack[x, :, :] = plt.imread(
                    tdf_data_folder + 'nf_%0.6d.tif' % (int(first_image) + x))
            else:
                counter += 1
        tdf = np.median(tdf_stack[counter:len(theta) - 1], axis=0)
        print('complete!')
        return tomopy.misc.corr.remove_neg(tdf, val=0)


def genBright(tbf_data_folder, tdf, tbf_fold_start=None, num2skip=None, tbf_num_imgs=None):
    if(tbf_fold_start is not None and num2skip is not None and tbf_num_imgs is not None):
        tbf_img_nums = np.arange(
            tbf_fold_start+num2skip, tbf_fold_start+num2skip+tbf_num_imgs, 1)
        tbf_num = len(tbf_img_nums)
        tbf_stack = np.zeros([tbf_num, 2048, 2048])
        print('Loading data for median bright field...')
        for ii in np.arange(tbf_num):
            tbf_stack[ii, :, :] = plt.imread(
                tbf_data_folder + 'nf_%0.6d.tif' % (tbf_img_nums[ii])) - tdf
            # image_stack[ii,:,:]=np.flipud(tmp_img>threshold)
        tbf = np.median(tbf_stack, axis=0)
        print('complete!')
        return tbf
    else:
        meta_data = _load_chess_metadata(tbf_data_folder)
        theta = meta_data[:, 1]
        first_image = sorted(os.listdir(tbf_data_folder))[0]
        first_image = first_image[3:first_image.index('.tif')]
        tbf_stack = np.zeros([len(theta), 2048, 2048])
        print('Loading data for median bright field...')
        counter = 0
        for x in range(len(theta) - 1):
            if theta[x] > 0:
                tbf_stack[x, :, :] = plt.imread(
                    tbf_data_folder + 'nf_%0.6d.tif' % (int(first_image) + x + 2)) - tdf
            else:
                counter += 1
        tbf = np.median(tbf_stack[counter:len(theta) - 1], axis=0)
        print('complete!')
        return tomopy.misc.corr.remove_neg(tbf, val=np.median(tbf) * 10)


def genTomo(tomoDataFolder, tdf, tbf, img_x_bounds, img_y_bounds, intCorr=None,
            tomo_fold_start=None, num2skip=None, tomo_num_imgs=None, theta=None):
    if tomo_fold_start is None:
        first_image = sorted(os.listdir(tomoDataFolder))[0]
        tomo_fold_start = int(first_image[3:len(first_image) - 4])
    if theta is None:
        meta_data = _load_chess_metadata(tomoDataFolder)
        theta = meta_data[:, 1]
    if num2skip is None:
        num2skip = _count_negative_theta_skip(theta)
    if tomo_num_imgs is None:
        tomo_num_imgs = len(theta) - num2skip

    tomo_img_nums = np.arange(
        tomo_fold_start + num2skip, tomo_fold_start + num2skip + tomo_num_imgs)
    x_slice = slice(img_x_bounds[0], img_x_bounds[1])
    y_slice = slice(img_y_bounds[0], img_y_bounds[1])
    tomo_imgs = np.zeros([
        1, tomo_num_imgs, img_x_bounds[1] - img_x_bounds[0],
        img_y_bounds[1] - img_y_bounds[0],
    ])

    if intCorr is None:
        intCorr = np.ones(tomo_num_imgs)

    tdf_crop = tdf[x_slice, y_slice]
    tbf_crop = tbf[x_slice, y_slice]
    print('Loading Images, Removing Negative Values, Applying Intensity Correction, Building Radiographs...')
    for ii in range(tomo_num_imgs):
        if ii % 100 == 0:
            print(f'Loading Image #: {ii}')
        tmp_img = plt.imread(tomoDataFolder + 'nf_%0.6d.tif' % tomo_img_nums[ii])
        tmp_img0 = tmp_img[x_slice, y_slice]
        tmp_img2 = tomopy.misc.corr.remove_neg(
            tmp_img0 - tdf_crop, val=0.0) * intCorr[ii]
        tomo_imgs[0, ii, :, :] = tomopy.prep.normalize.minus_log(tmp_img2 / tbf_crop)
    print("complete!")
    return np.swapaxes(tomo_imgs, 1, 2)


def getIntCorr(tomoDataFolder):
    """Return normalized intensity correction values for a CHESS tomography folder."""
    meta_data = _load_chess_metadata(tomoDataFolder)
    theta = meta_data[:, 1]
    num2skip = _count_negative_theta_skip(theta)
    tomo_num_imgs = len(theta) - num2skip
    ic = meta_data[_chess_scan_slice(meta_data, num2skip, tomo_num_imgs), 6]
    return ic / np.median(ic)


def getTheta(tomoDataFolder):
    """Return theta values in radians for a CHESS tomography folder."""
    meta_data = _load_chess_metadata(tomoDataFolder)
    theta = meta_data[:, 1]
    num2skip = _count_negative_theta_skip(theta)
    tomo_num_imgs = len(theta) - num2skip
    return np.deg2rad(meta_data[_chess_scan_slice(meta_data, num2skip, tomo_num_imgs), 1])
