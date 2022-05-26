# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 09:08:56 2018

@author: jk989
"""
from turtle import end_fill
import tomopy
import h5py
import numpy as np
import scipy.ndimage as img
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib import animation
import skimage
import os

#####RECONSTRUCTION FUNCTIONS####


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
    f = h5py.File(fileName, 'r')
    tomoImgs = f[sinogramName][:]
    theta = f[thetaName][:]
    return [tomoImgs, theta]


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
    if(len(image) == 1):
        return _guessVBounds(image[0], area)
    # only uses middle section of image

    image = image[len(image)//4:len(image)*3//4,
                  len(image[0])//4:len(image[0])*3//4]
    flatten = image.flatten()
    a = np.ma.masked_invalid(flatten)
    flatten = np.nan_to_num(flatten)
    n, bins = np.histogram(flatten, 100, range=(a.min(), a.max()))
    counted = 0.
    total = sum(n)
    tempBins = []
    tempN = []
    for x in range(len(n)):
        if n[x] / float(total) > .005:
            tempBins += [bins[x]]
            tempN += [n[x]]
    bins = tempBins
    n = tempN
    initMin = 0
    initMax = 0
    for x in range(len(bins)):
        if x % 2 == 0 and counted / total < 1 - area:
            counted += n[x/2]
            initMin += 1
        elif x % 2 == 1 and counted / total < 1 - area:
            counted += n[len(bins) - x/2 - 2]
            initMax += 1
        else:
            break
    initMin = bins[initMin]
    initMax = bins[len(bins) - initMax - 1]
    if(initMin >= initMax):
        initMin, initMax = 0, 1
        print("Couldn't generate vmin/vmax bounds with given inputs!")

    #print(str(initMin)+ " " + str(initMax))
    return [initMin, initMax]


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
    if(num >= 0):
        return str(round(10 ** np.ceil(-1*np.log10(num))*num, 1)) + 'e' + str(int(np.floor(np.log10(num))))
    else:
        num = -num
        return '-' + _toSci(num)


def reconstruct(sinograms, centers, imageBounds, layers, theta, sigma=.1, ncore=4, algorithm='gridrec', run_secondary_sirt=False, secondary_iter=100):
    """
    Reconstructs object from projection data.
    Takes in list [elements, rows, angles, cols]
    or [rows, angles, cols],
    and returns ndarray representing a 3D reconstruction

    Parameters
    ----------
    sinograms : ndarray
        3D tomographic data.
    centers : scalar, array
        estimated location(s) of rotation axis relative to imageBounds
    imageBounds : len 2 array
        boundary of sample to be evaluated
    layers : scalar, len 2 array
        single layer or bounds of layers
    theta : ndarray
        list of angles used in tomography data (radians)
    sigma : float
        damping param in Fourier space
    ncore : int
        # of cores that will be assigned
    algorithm : {str, function}
        see tomopy.recon.algorithm for list of algs

    Returns
    -------
    ndarray
        Reconstructed multi-elemental 3D object.
    """
    # normalizing recon params
    if np.isscalar(layers):
        layers = [layers, layers+1]
    if np.isscalar(centers):
        centers = np.full(len(layers)-1, centers)
    elif len(centers) == 2:
        assert len(layers) >= 2, "Too many centers for number of layers!"
        arr = np.zeros(layers[1]-layers[0])
        for x in range(layers[1]-layers[0]):
            arr[x] += centers[0] + (centers[1]-centers[0]) / \
                float(layers[1]-layers[0])*x
        centers = arr
    else:
        assert len(centers) == layers[1] - \
            layers[0], "unequal # of centers and layers!"
    # begin reconstruction
    if len(sinograms.shape) == 4:
        sinograms = np.swapaxes(sinograms, 1, 2)
        recons = np.zeros([len(sinograms), layers[1]-layers[0],
                           imageBounds[1]-imageBounds[0], imageBounds[1]-imageBounds[0]])
        recon_clean = np.zeros([len(sinograms), layers[1]-layers[0],
                                imageBounds[1]-imageBounds[0], imageBounds[1]-imageBounds[0]])
        for el in range(len(sinograms)):
            print("Process Element #" + str(el))
            for x in range(layers[1]-layers[0]):
                # if x %1 == 0:
                print("Processing Layer #" + str(x))
                tmp = tomopy.prep.stripe.remove_stripe_fw(
                    sinograms[el, :, layers[0]+x:layers[0]+x+1, imageBounds[0]:imageBounds[1]], sigma=sigma, ncore=ncore)

                tmp_recon = tomopy.recon(tmp, theta, center=(
                    imageBounds[1] - imageBounds[0])/2.0 + centers[x], algorithm=algorithm, sinogram_order=False, ncore=ncore)

                if run_secondary_sirt:
                    options = {'proj_type': 'cuda',
                               'method': 'SIRT_CUDA', 'num_iter': secondary_iter}
                    recon = tomopy.recon(tmp, theta, center=(
                        imageBounds[1] - imageBounds[0])/2.0 + centers[x], init_recon=tmp_recon, algorithm=tomopy.astra, options=options, sinogram_order=False, ncore=ncore)
                    recons[el][x] += recon[0]
                else:
                    recons[el][x] += tmp_recon
        recon_clean[el, :] = tomopy.misc.corr.remove_ring(
            recons[el, :], rwidth=17)

    # if there is no element dimension
    elif len(sinograms.shape) == 3:
        sinograms = np.swapaxes(sinograms, 0, 1)
        recons = np.zeros([1, layers[1]-layers[0], imageBounds[1] -
                           imageBounds[0], imageBounds[1]-imageBounds[0]])
        recon_clean = np.zeros(
            [1, layers[1]-layers[0], imageBounds[1]-imageBounds[0], imageBounds[1]-imageBounds[0]])
        for x in range(layers[1]-layers[0]):
            # if x %1 == 0:
            print("Process Layer #" + str(x))
            tmp = tomopy.prep.stripe.remove_stripe_fw(
                sinograms[:, layers[0]+x:layers[0]+x+1, imageBounds[0]:imageBounds[1]], sigma=sigma, ncore=ncore)
            tmp_recon = tomopy.recon(tmp, theta, center=(
                imageBounds[1] - imageBounds[0])/2.0 + centers[x], algorithm=algorithm, sinogram_order=False, ncore=ncore)

            if run_secondary_sirt:
                options = {'proj_type': 'cuda',
                           'method': 'SIRT_CUDA', 'num_iter': secondary_iter}
                recon = tomopy.recon(tmp, theta, center=(imageBounds[1] - imageBounds[0])/2.0 + centers[x],
                                     init_recon=tmp_recon, algorithm=tomopy.astra, options=options, sinogram_order=False, ncore=ncore)
                recons[x] += recon[0]
            else:
                recons[x] += tmp_recon

        recon_clean[0, :] = tomopy.misc.corr.remove_ring(
            recons[0, :], rwidth=17)

    if(layers[1]-layers[0] != 1):
        print("complete!")
    return recon_clean


#	    if algorithm =='gridrec':
#	        tmp_recon = tomopy.recon(tmp, theta, center = (imageBounds[1] - imageBounds[0])/2.0 + centers[x], algorithm=algorithm,sinogram_order=False,ncore=ncore)
#	    if algorithm =='FBP_CUDA':
#            tmp_recon = tomopy.recon(tmp, theta, center= (imageBounds[1] - imageBounds[0])/2.0 + centers[x], algorithm=tomopy.astra, options={'method': 'FBP_CUDA', 'proj_type': 'cuda'},sinogram_order=False,ncore=24)


class ValHelper:
    """
    Class to help user find reconstruction vals
    """

    def __init__(self, fig2, sinograms, imageBounds, layer, layerBounds, theta, sigma, ncore, algorithm, vmin, vmax, cmap, interpolation):
        """
        initalizes vals and launches GUI
        """
#        self.fig1 = fig1
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

        self.element = 0
        self.recon = reconstruct(sinograms[self.element], self.center, imageBounds,
                                 self.layer + self.layerBounds[0], theta, sigma, ncore, algorithm)
        if vmin is not None and vmax is not None:
            self.vMin = vmin
            self.vMax = vmax
        else:
            vBounds = _guessVBounds(self.recon[0])
            self.vMin = vBounds[0]
            self.vMax = vBounds[1]
        self.cmap = cmap
        self.interpolation = interpolation


#        self.fig1.imshow(img.filters.gaussian_filter(self.recon[0][0],0.75), vmin = self.vMin, vmax = self.vMax, cmap = self.cmap)
#        self.fig1.cla() #to keep in shape
        self.fig2.imshow(img.filters.gaussian_filter(
            self.recon[0][0], 0.75), vmin=self.vMin, vmax=self.vMax, cmap=self.cmap)
#        self.fig1.set_title("Previous Version")
        #self.fig2.set_title("Current Version")
        self.prevMin = self.vMin
        self.prevMax = self.vMax

        # define axes
        axcolor = 'lightgoldenrodyellow'
        axMax = plt.axes([0.08, 0.04, 0.65, 0.025], facecolor=axcolor)
        axMin = plt.axes([0.08, 0.08, 0.65, 0.025], facecolor=axcolor)
        axCenter = plt.axes([0.08, 0.12, 0.65, 0.025], facecolor=axcolor)
        axLayer = plt.axes([0.08, .16, .09, 0.025])
        axElement = plt.axes([0.28, .16, .09, 0.025])
        axCenterButton = plt.axes([0.46, .16, .14, 0.025])

        # define prompter
        self.prompt = plt.text(1.1, 0, "Ready.")

        # define sliders
        self.minSlide = Slider(
            axMin, 'vMin', self.vMin, self.vMin + (self.vMax-self.vMin)/2, valinit=self.vMin)
        self.maxSlide = Slider(
            axMax, 'vMax', self.vMin + (self.vMax-self.vMin)/2, self.vMax, valinit=self.vMax)
        self.centerSlide = Slider(axCenter, 'center', max(-200, (imageBounds[0]-imageBounds[1])/2), min(
            200, (imageBounds[1]-imageBounds[0])/2), valinit=0.00)
        self.minSlide.valtext.set_text(_toSci(self.minSlide.val))
        self.maxSlide.valtext.set_text(_toSci(self.maxSlide.val))
        self.minSlide.on_changed(self.smallUpdate)
        self.maxSlide.on_changed(self.smallUpdate)
        self.centerSlide.on_changed(self.largeUpdate)

        # define textboxs
        self.layerButton = TextBox(
            axLayer, "layer ", color=axcolor, initial=str(self.layer))
        self.layerButton.on_submit(self.layerSubmit)
        self.elementButton = TextBox(
            axElement, "element ", color=axcolor, initial="0")
        self.elementButton.on_submit(self.elementSubmit)
        self.centerButton = TextBox(
            axCenterButton, "center ", color=axcolor, initial="0")
        self.centerButton.on_submit(self.centerSubmit)

        # define undo
        axUndo = plt.axes([.85, .05, .09, .06])
        self.undoButton = Button(
            axUndo, 'Undo', color=axcolor, hovercolor='0.975')
        self.undoButton.on_clicked(self.undo)
        self.versions = [[self.center, self.imageBounds,
                          self.layer, self.vMin, self.vMax, self.element]]

    def largeUpdate(self, val):
        """
        Called on change to center slider val. Changes center of recon and plots
        """
        self.center = self.centerSlide.val
#        self.fig1.imshow(img.filters.gaussian_filter(self.recon[0][0],0.75), vmin = self.vMin, vmax = self.vMax, cmap = self.cmap, interpolation = self.interpolation)
        self.recon = reconstruct(self.sinograms[self.element], self.center, self.imageBounds,
                                 self.layer + self.layerBounds[0], self.theta, self.sigma, self.ncore, self.algorithm)
        self.fig2.imshow(img.filters.gaussian_filter(
            self.recon[0][0], 0.75), vmin=self.vMin, vmax=self.vMax, cmap=self.cmap, interpolation=self.interpolation)
        self.versions += [[self.center, self.imageBounds,
                           self.layer, self.vMin, self.vMax, self.element]]
        self.prompt.set_text("Center Changed.")
        # self.centerSubmit.set_text(self.center)

    def smallUpdate(self, val):
        """
        Called on change to vmin/vmax slider val. Changes vmin/vmax.
        Does not recalculate self.recon
        """
#        self.fig1.imshow(img.filters.gaussian_filter(self.recon[0][0],0.75), vmin = self.vMin, vmax = self.vMax, cmap = self.cmap, interpolation = self.interpolation)
        self.vMin = self.minSlide.val
        self.vMax = self.maxSlide.val
        self.fig2.imshow(img.filters.gaussian_filter(
            self.recon[0][0], 0.75), vmin=self.minSlide.val, vmax=self.maxSlide.val, cmap=self.cmap, interpolation=self.interpolation)
        self.minSlide.valtext.set_text(_toSci(self.minSlide.val))
        self.maxSlide.valtext.set_text(_toSci(self.maxSlide.val))
        self.prompt.set_text("vMin/vMax Changed.")

        self.versions += [[self.center, self.imageBounds,
                           self.layer, self.vMin, self.vMax, self.element]]

    def layerSubmit(self, text):
        """
        Called on input to layer textbox. Changes layer of recon and plots
        """
        try:
            self.layer = int(text)
            if(self.layer in range(0, self.layerBounds[1]-self.layerBounds[0])):
                #                self.fig1.imshow(img.filters.gaussian_filter(self.recon[0][0],0.75)], vmin = self.minSlide.val, vmax = self.maxSlide.val, cmap = self.cmap, interpolation = self.interpolation)
                self.recon = reconstruct(self.sinograms[self.element], self.center, self.imageBounds,
                                         self.layer + self.layerBounds[0], self.theta, self.sigma, self.ncore, self.algorithm)
                self.fig2.imshow(img.filters.gaussian_filter(
                    self.recon[0][0], 0.75), vmin=self.vMin, vmax=self.vMax, cmap=self.cmap, interpolation=self.interpolation)
                self.versions += [[self.center, self.imageBounds,
                                   self.layer, self.vMin, self.vMax, self.element]]
                self.prompt.set_text("Layer Changed.")
            else:
                self.prompt.set_text("Invalid Layer!")
        except:
            self.prompt.set_text("Not an Integer!")

    def elementSubmit(self, text):
        """
        Called on input to element textbox. Changes element of recon and plots
        """
        try:
            el = int(text)
            if(el in range(0, len(self.sinograms))):
                self.element = el
#                self.fig1.imshow(img.filters.gaussian_filter(self.recon[0][0],0.75), vmin = self.minSlide.val, vmax = self.maxSlide.val, cmap = self.cmap, interpolation = self.interpolation)
                self.recon = reconstruct(self.sinograms[self.element], self.center, self.imageBounds,
                                         self.layer + self.layerBounds, self.theta, self.sigma, self.ncore, self.algorithm)
                self.fig2.imshow(img.filters.gaussian_filter(
                    self.recon[0][0], 0.75), vmin=self.vMin, vmax=self.vMax, cmap=self.cmap, interpolation=self.interpolation)
                self.versions += [[self.center, self.imageBounds,
                                   self.layer, self.vMin, self.vMax, self.element]]
                self.prompt.set_text("Element Changed.")
            else:
                self.prompt.set_text("Invalid Element!")
        except:
            self.prompt.set_text("Not an Integer!")

    def centerSubmit(self, text):
        """
        Called on input to center textbox. Changes element of recon and plots
        """
        isPos = True
        try:
            if(text[0:4] == 'neg(' and text[-1] == ')'):
                isPos = False
                text = text[4:-1]
            if isPos:
                self.center = float(text)
            else:
                self.center = -float(text)
            self.centerSlide.set_val(self.center)
        except:
            self.prompt.set_text("Not an scalar!")

    def undo(self, val):
        """
        Called on undo button press. Recalls previous version and plots
        """
        if(len(self.versions) >= 3):
            newVer = self.versions[-3]
            tempRecon = reconstruct(self.sinograms[self.element], newVer[0], newVer[1],
                                    newVer[2], self.theta, self.sigma, self.ncore, self.algorithm)
            self.fig1.imshow(img.filters.gaussian_filter(
                tempRecon[0][0], 0.75), vmin=newVer[3], vmax=newVer[4], cmap=self.cmap, interpolation=self.interpolation)
        else:
            #            self.fig1.cla()
            self.fig1.set_title("Previous Version")
        if(len(self.versions) >= 2):
            oldVer = self.versions[-2]
            self.recon = reconstruct(self.sinograms[self.element], oldVer[0], oldVer[1],
                                     oldVer[2], self.theta, self.sigma, self.ncore, self.algorithm)
            self.fig2.imshow(img.filters.gaussian_filter(
                self.recon[0][0], 0.75), vmin=oldVer[3], vmax=oldVer[4], cmap=self.cmap, interpolation=self.interpolation)
            self.versions = self.versions[:len(self.versions)-1]
            self.center = oldVer[0]
            self.imageBounds = oldVer[1]
            self.layer = oldVer[2]
            self.vMin = oldVer[3]
            self.vMax = oldVer[4]
            self.element = oldVer[5]
            self.prompt.set_text("Undo Complete.")
        else:
            self.prompt.set_text("Nothing to Undo!")

    def getCenter(self):
        """
        returns center
        """
        return self.center

    def getLayer(self):
        """
        returns layer
        """
        return self.layer + self.layerBounds[0]


def launchValHelper(sinograms, imageBounds, layer, layerBounds, theta, sigma=.1, ncore=4, algorithm='gridrec', vmin=None, vmax=None, cmap='gray', interpolation='none'):
    """
    Launches GUI which allows users to pick reconstruction variables

    Parameters
    ----------
    sinograms : 4D ndarray
        3D reconstruction data [elements, rows, angles, cols]
    imageBounds : len 2 array
        boundary of sample to be evaluated
    layer : scalar
        initial layer to be displayed
    theta : ndarray
        list of angles used in tomography data (radians)
    sigma : float
        damping param in Fourier space
    ncore : int
        # of cores that will be assigned
    algorithm : {str, function}
        see tomopy.recon.algorithm for list of algs
    vmin/vmax : scalar
        lower/upper bound of color spectrum
    cmap : Colormap (matplotlib.colors.Colormap)
        function which takes in scalars and outputs colors (determines color scheme)
    interpolation : string
        determines algo for interpolation, see matplotlib.pyplot.imshow for list of algos

    Returns
    -------
    instance of ValHelper
        instance which holds reconstruction vals
    """
    plt.close()
#    fig1 = plt.axes([.05,.25,.43,.6])
#    fig2 = plt.axes([.05+0.51,.25,.43,.6])
    fig2 = plt.axes([.05, .25, .7, .7])
    return ValHelper(fig2, sinograms, imageBounds, layer, layerBounds, theta, sigma, ncore, algorithm, vmin, vmax, cmap, interpolation)


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
    """
    Takes the difference between topCenter and bottomCenter, assumes
    center movement is linear, and calculates a list of centers for
    layerBounds.

    Parameters
    ----------
    layersBounds : len 2 array
        reconstruction layer bounds
    topLayer/bottomLayer : int
        layers from top and bottom of stack with known centers
    topCenter/bottomCenter : scalar
        center vals for topLayer/bottomLayer

    Returns
    -------
    array containing extrapolated centers for layerBounds    
    """
#    arr = np.zeros(layerBounds[1]-layerBounds[0])
#    slope = (topCenter - bottomCenter)/(float(topLayer-bottomLayer))
#    inital = topCenter + slope*(topLayer-layerBounds[0])
#    for x in range(layerBounds[1]-layerBounds[0]):
#        arr[x] += inital - slope*x
#    return arr

    arr = np.linspace(topCenter, bottomCenter,
                      num=layerBounds[1]-layerBounds[0])
    return arr


def rotate(imgs, theta):
    """
    rotates an image

    Parameters
    ----------
    imgs : ndarray
        images to rotate
    theta : scalar
        rotation val (counterclockwise, DEGREES)

    Returns
    -------
    ndarray
        rotated image
    """
    return skimage.transform.rotate(img, np.degrees(theta))


def crop(imgs, left, right, top, bottom):
    """
    rotates an image

    Parameters
    ----------
    imgs : ndarray
        images to crop
    left/right/top/bottom : int
        pixel vals of desired boundaries

    Returns
    -------
    ndarray
        cropped image
    """
    return skimage.util.crop(img, [(top, len(img[:][0]) - bottom), (left, len(img[0]) - right)])


def save(filename, names, data):
    """
    saves files as hdf5 in specified directory

    Parameters
    ----------
    fileName : String
        desired name of file
    names : [string,]
        list of desired names of data
    data : [ndarray,]
        list of data
    datatype : type
        desired datatype

    Returns
    -------
    None
    """
#    if data[0].dtype != datatype:
#        for x in range(len(data)):
#            data[x] = data[x].astype(datatype)
    hf = h5py.File(str(filename), 'w')
    try:
        for x in range(len(data)):
            hf.create_dataset(names[x], data=data[x])
        hf.close()
        print("File saved at " + str(filename))
    except:
        hf.close()
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
        """
        updates figure
        """
        self.index = (self.index + 1) % self.volume.shape[0]
        #bounds = _guessVBounds(self.volume[self.index])
        self.im.set_array(self.volume[self.index])
        self.ax.set_title('Image %d' % (self.index))
        #self.im.set_clim(bounds[0], bounds[1])
        return self.im, self.ax,


def multiSliceGiffer(volume, cmap=None):
    """
    display that cycles through slices of a volume
    """
    Giffer(volume, cmap)

# BAD


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
    else:  # optional params must be calculated
        # finding the meta data
        folder = os.listdir(
            tdf_data_folder[:len(tdf_data_folder)-3] + 'scalars')
        metaData = np.loadtxt(tdf_data_folder[:len(
            tdf_data_folder)-3] + 'scalars/' + folder[0] + '/summary.dat')
        theta = metaData[:, 1]
        firstImage = sorted(os.listdir(tdf_data_folder))[0]
        firstImage = firstImage[3:firstImage.index('.tif')]
        tdf_stack = np.zeros([len(theta), 2048, 2048])

        print('Loading data for median dark field...')
        counter = 0
        for x in range(0, len(theta)-1):
            if(theta[x] > 0):
                # print(tdf_data_folder+'nf_%0.6d.tif' %(int(firstImage) + x))
                tdf_stack[x, :, :] = plt.imread(
                    tdf_data_folder+'nf_%0.6d.tif' % (int(firstImage) + x))
            else:
                counter += 1    
        tdf = np.median(tdf_stack[counter:len(theta)-1], axis=0)
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
        folder = os.listdir(
            tbf_data_folder[:len(tbf_data_folder)-3] + 'scalars')
        metaData = np.loadtxt(tbf_data_folder[:len(
            tbf_data_folder)-3] + 'scalars/' + folder[0] + '/summary.dat')
        theta = metaData[:, 1]
        firstImage = sorted(os.listdir(tbf_data_folder))[0]
        firstImage = firstImage[3:firstImage.index('.tif')]
        tbf_stack = np.zeros([len(theta), 2048, 2048])
        print('Loading data for median bright field...')
        counter = 0
        for x in range(0, len(theta)-1):
            if(theta[x] > 0):
                tbf_stack[x, :, :] = plt.imread(
                    tbf_data_folder+'nf_%0.6d.tif' % (int(firstImage) + x + 2)) - tdf
                # image_stack[x,:,:]=np.flipud(tmp_img>threshold)
            else:
                counter += 1
        tbf = np.median(tbf_stack[counter:len(theta)-1], axis=0)
        print('complete!')
        return tomopy.misc.corr.remove_neg(tbf, val=np.median(tbf)*10)


def genTomo(tomoDataFolder, tdf, tbf, img_x_bounds, img_y_bounds, intCorr=None, tomo_fold_start=None, num2skip=None, tomo_num_imgs=None, theta=None):
    if(tomo_fold_start is None):
        tomo_fold_start = sorted(os.listdir(tomoDataFolder))[0]
        tomo_fold_start = int(tomo_fold_start[3:len(tomo_fold_start)-4])
    if(tomoDataFolder is None):
        folder = os.listdir(tomoDataFolder[:len(tomoDataFolder)-3] + 'scalars')
        metaData = np.loadtxt(tomoDataFolder[:len(
            tomoDataFolder)-3] + 'scalars/' + folder[0] + '/summary.dat')
    if(theta is None):
        theta = metaData[:, 1]
    if(num2skip is None):
        num2skip = 2
        for x in range(len(theta)):
            if theta[x] < 0:
                num2skip += 1
            else:
                break
    if(tomo_num_imgs is None):
        tomo_num_imgs = len(theta) - num2skip

    tomo_img_nums = np.arange(tomo_fold_start+num2skip,
                              tomo_fold_start+num2skip+tomo_num_imgs, 1)
    tomoImgs = np.zeros([1, tomo_num_imgs, (img_x_bounds[1] -
                                            img_x_bounds[0]), (img_y_bounds[1]-img_y_bounds[0])])

    if(intCorr is None):
        intCorr = np.ones(tomo_num_imgs)
    # k
    num_tomoImgs = tomo_num_imgs
    # numbers for intensity corrections values of ic0/meidan of ic0
    print('Loading Images, Removing Negative Values, Applying Intensity Correction, Building Radiographs...')
    for ii in np.arange(num_tomoImgs):
        if(ii % 100 == 0):
            print('Loading Image #: ' + str(ii))
        # loads data
        tmp_img = plt.imread(tomoDataFolder+'nf_%0.6d.tif' %
                             (tomo_img_nums[ii]))
        # removes negative and applies intensity corrections
        tmp_img0 = tmp_img[img_x_bounds[0]:img_x_bounds[1],
                           img_y_bounds[0]:img_y_bounds[1]]
        tmp_img2 = tomopy.misc.corr.remove_neg(
            tmp_img0-tdf[img_x_bounds[0]:img_x_bounds[1], img_y_bounds[0]:img_y_bounds[1]], val=0.0,)*intCorr[ii]

        # normalizes in some way idk
        tomoImgs[0, ii, :, :] = tomopy.prep.normalize.minus_log(
            tmp_img2/tbf[img_x_bounds[0]:img_x_bounds[1], img_y_bounds[0]:img_y_bounds[1]])
    print("complete!")
    return np.swapaxes(tomoImgs, 1, 2)


def getIntCorr(tomoDataFolder):
    """
    Finds intensity correction values for given tomographic data folder

    Parameters
    ----------
    tomoDataFolder : String
        name of CHESS tomographic data folder

    Returns
    -------
    ndarray
        contains intensity correction values for tomographic data

    WARNINGS
    --------
    only use to parse CHESS data.
    """
    tomo_fold_start = sorted(os.listdir(tomoDataFolder))[0]
    tomo_fold_start = int(tomo_fold_start[3:len(tomo_fold_start)-4])
    folder = os.listdir(tomoDataFolder[:len(tomoDataFolder)-3] + 'scalars')
    metaData = np.loadtxt(tomoDataFolder[:len(
        tomoDataFolder)-3] + 'scalars/' + folder[0] + '/summary.dat')
    theta = metaData[:, 1]
    num2skip = 2
    for x in range(len(theta)):
        if theta[x] < 0:
            num2skip += 1
        else:
            break
    tomo_num_imgs = len(theta) - num2skip
    return metaData[(num2skip-2):(num2skip+tomo_num_imgs-2), 6]/np.median(metaData[(num2skip-2):(num2skip+tomo_num_imgs-2), 6])


def getTheta(tomoDataFolder):
    """
    Finds theta values for given tomographic data folder

    Parameters
    ----------
    tomoDataFolder : String
        name of CHESS tomographic data folder

    Returns
    -------
    ndarray
        contains theta values for tomographic data

    WARNINGS
    --------
    only use to parse CHESS data.
    """
    tomo_fold_start = sorted(os.listdir(tomoDataFolder))[0]
    tomo_fold_start = int(tomo_fold_start[3:len(tomo_fold_start)-4])
    folder = os.listdir(tomoDataFolder[:len(tomoDataFolder)-3] + 'scalars')
    metaData = np.loadtxt(tomoDataFolder[:len(
        tomoDataFolder)-3] + 'scalars/' + folder[0] + '/summary.dat')
    theta = metaData[:, 1]
    num2skip = 2
    for x in range(len(theta)):
        if theta[x] < 0:
            num2skip += 1
        else:
            break
    tomo_num_imgs = len(theta) - num2skip
    return np.deg2rad(metaData[(num2skip-2):(num2skip+tomo_num_imgs-2), 1])
