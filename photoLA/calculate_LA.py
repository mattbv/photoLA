# -*- coding: utf-8 -*-
"""
Module with all functions required to estimate Leaf Area. These
functions are designed to be imported into another script ot use.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""
import numpy as np
import cv2
from sklearn.cluster import KMeans
from win32api import GetSystemMetrics


def resize_img(img):

    """
    Function to resize an image in case it is larger than main
    display resolution.

    Parameters
    ----------
    img: array
        RGB image as an MxNx3 array.

    Returns
    -------
    res: array
        Resized RGB image as an (M * f)x(N * f)x3 array.

    """

    # Obtains image and display width (w) and height (h) resolutions.
    img_h, img_w, _ = img.shape
    disp_w = GetSystemMetrics(0) - 100
    disp_h = GetSystemMetrics(1) - 100

    # Calculate rations between display and image resolutions.
    ratio_h = disp_h / float(img_h)
    ratio_w = disp_w / float(img_w)

    # Check if image is larger than display and if so, resize image.
    if (ratio_h <= 1) or (ratio_w <= 1):

        # Calculating resizing factor by selecting the smaller ration, which
        # means, the dimension with larger excess over display resolution.
        factor = np.min([ratio_h, ratio_w])

        # Performing resizing.
        res = cv2.resize(img, None, fx=factor, fy=factor,
                         interpolation=cv2.INTER_CUBIC)
    else:
        # If image is already smaller than display, return it without
        # changes.
        res = img

    return res


def max_rgb_filter(image):

    """
    Function to perform a channel maximization by enhancing the one with
    larger intensity from all 3 channels.
    This function is presented and thoroughly explained by Adrian Rosebrock at
    [1].

    Parameters
    ----------
    image: array
        RGB image as an MxNx3 array.

    Returns
    -------
    image: array
        RGB image as an MxNx3 processed array.

    Reference
    ---------
    .. [1] http://www.pyimagesearch.com/2015/09/28/implementing-the-max-rgb-\
filter-in-opencv/

    """

    # Split the image into its BGR components.
    (B, G, R) = cv2.split(image)

    # Find the maximum pixel intensity values for each
    # (x, y)-coordinate,, then set all pixel values less
    # than M to zero.
    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    # Merge the channels back together and return the image.
    return cv2.merge([B, G, R])


def calculate_area(img_leaf, img_ref, area_ref):

    """
    Function to calculate the area of a given object in a masked image by
    applying a direct relationship to another object with known area.
    The relationship is based on the number of masked pixels of target and
    reference objects.

    Parameters
    ----------
    img_leaf: array
        Masked image for the leaf image. Pixels that are not leaves should have
        a value of 0.
    img_ref: array
        Masked image for the reference image. Pixels that are not reference
        should have a value of 0.
    area_ref: scalar
        Known area of the reference object.

    Returns
    -------
    leaf_area: scalar
        Estimated leaf area in the same units as area_ref.

    """

    # Counting number of non-zero pixels in both img_leaf and img_ref.
    n_pixels_leaf = count_pixels(img_leaf)
    n_pixels_ref = count_pixels(img_ref)

    # Calculating leaf area.
    leaf_area = (n_pixels_leaf * area_ref) / n_pixels_ref

    return leaf_area


def process_img(im):

    """
    Function used to perform image processing steps necessary to leaf
    area calculation.

    Parameters
    ----------
    im: array
        RGB image as an MxNx3 array.

    Returns
    -------
    leaf_pixels: array
        RGB image of masked leaf pixels as an MxNx3 array.
    ref_pixels: array
        RGB image of masked reference pixels as an MxNx3 array.

    """

    # Checking image resolution and, if necessary, resizing it to fit screen.
    im = resize_img(im)

    # Setting parameter fromCenter to False. This parameter controls if the
    # user input rectangles should grown from their center (if True) or from
    # upper-left corner (if False).
    fromCenter = False

    # Calling input from the user. This spawns the target image and waits for
    # the user to draw a rectangle to select the area to be cropped.
    # First instance, crop the image to select the reference area.
    r = cv2.selectROI(im, fromCenter)
    imRef = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # Second instance, crop the image to select the leaf area.
    r = cv2.selectROI(im, fromCenter)
    imLeaf = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # Close image window after selecting ROIs.
    cv2.destroyWindow("ROI selector")

    # Pre-processing the leaf ROI image to increase contrast.
    imLeaf_hc = increase_contrast(imLeaf)

    # Processing leaf image keep only leaf pixels.
    leaf_pixels = get_leaf(imLeaf_hc)
    # Processing reference image to keep only reference pixels
    ref_pixels = get_ref(imRef)

    return leaf_pixels, ref_pixels


def increase_contrast(img):

    """
    Function to perform contrast enhancing on an image. This code was presented
    originaly by Jeru Luke in [1].

    Parameters
    ----------
    img: array
        RGB image as an MxNx3 array.

    Return
    ------
    final: array
        Contrast enhanced RGB image as an MxNx3 array.

    Reference
    ---------
    .. [1]  https://stackoverflow.com/questions/39308030/how-do-i-increase\
-the-contrast-of-an-image-in-python-opencv

    """

    # Converting image from RGB to LAB.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to different channels.
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel.
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to RGB model.
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final


def get_channel_class(img, c):

    """
    Function to mask out an RGB image based on values from one of its three
    channels. The function masks out zero value pixels on the selected channel.

    Parameters
    ----------
    img: array
        RGB image as an MxNx3 array.
    c: scalar
        Channel (R[0], G[1] or B[2]) to filter.


    Returns
    -------
    res: array
        Resulting RGB image as an MxNx3 array.


    """

    # Maxing out each pixels channel with highest intensity.
    im = max_rgb_filter(img)
    # Splitting image into channels.
    channels = cv2.split(im)

    # Applying bitwise mask to filter out pixels with zero value on channel c.
    res = cv2.bitwise_and(img, img, mask=channels[c])

    return res


def get_leaf(img, stdev_threshold=5):

    """
    Function to mask leaf pixels from an image.

    Parameters
    ----------
    img: array
        RGB image as an MxNx3 array.
    stdec_threshold: scalar
        Standard deviation threshold to remove background pixels.

    Returns
    -------
    leaf_img: array
        RGB image of masked leaf pixels as an MxNx3 array.

    """

    # Obtaining width and height (in pixels) of the input image.
    h, w, _ = img.shape

    # Denoising input RGB image.
    dst = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    # Calculating pixelwise standard deviation of the input image.
    stdev = np.std(img, axis=2)

    # Performing leaf masking by channel values.
    leaf_channel = get_channel_class(dst, 1)

    # Masking image based on standard deviation threshold.
    mask = np.zeros([h, w], dtype=np.uint8)
    mask[stdev >= stdev_threshold] = 1

    # Masking extracted leaf image using standard deviation mask.
    masked = cv2.bitwise_and(leaf_channel, leaf_channel, mask=mask)

    # Closing possible gaps within leaf pixels in the masked leaf image.
    leaf_img = close_gaps(masked)

    return leaf_img


def get_ref(im):

    """
    Function to mask reference material pixels from an image.

    Parameters
    ----------
    im: array
        RGB image as an MxNx3 array.

    Returns
    -------
    res: array
        Reference material image as an MxNx3 array.

    """

    # Obtaining width and height (in pixels) of the input image.
    h, w, _ = im.shape

    # Extracting RGB channels.
    imB = im[:, :, 0]
    imG = im[:, :, 1]
    imR = im[:, :, 2]

    # Converting image arrays into vectors.
    imB_V = np.reshape(imB, [1, h * w])
    imG_V = np.reshape(imG, [1, h * w])
    imR_V = np.reshape(imR, [1, h * w])

    # Stacking vectors as an (M*N)x3 array. In this array, each row represents
    # a different pixel.
    im_V = np.vstack((imR_V, imG_V, imB_V)).T

    # Performing k-means clustering with 2 clusters.
    kmeans = KMeans(n_clusters=2, n_jobs=-1).fit(im_V)

    # Obtaining classes labels and reshaping them back to the original MxN
    # image resolution.
    classes = kmeans.labels_
    classes = classes.reshape([h, w])

    # Detecting center class (which by the ROI selection should be the
    # reference class).
    center_class = classes[int(h / 2), int(w / 2)]

    # Masking all pixels that are in the same class as the center pixel.
    mask = np.zeros([h, w], dtype=np.uint8)
    mask[classes == center_class] = 1

    # Masking input image using center class masked array as mask.
    res = cv2.bitwise_and(im, im, mask=mask)

    return res


def close_gaps(img, k=3, n_iter=5):

    """
    Function to apply two iterative process to dilate and then erode the
    input RGB image. The aim of this function is to close gaps (or isolated
    clusters of pixels) in an RGB image.

    Parameters
    ----------
    img: array
        RGB image as an MxNx3 array.
    k: scalar
        Kernel size used in both dilation and erosion steps. The larger k is,
        the greater is the smoothing effects over the input image.
        Default is 3.
    n_iter: scalar
        Number of iterations to run for each step. Default is 5.

    Returns
    -------
    img_erosion: array
        Processed RGB image as an MxNx3 array.

    """

    # Creating the k-by-k kernel.
    kernel = np.ones((k, k), np.uint8)

    # Performing image dilation and the image erosion.
    img_dilation = cv2.dilate(img, kernel, iterations=n_iter)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=n_iter)

    return img_erosion


def count_pixels(img):

    """
    Function to calculate the number of non-zero pixels in an image.

    Parameters
    ----------
    img: array
        RGB image as an MxNx3 array.

    Returns
    -------
    count: scalar
        Total number of non-zero pixels.

    """

    # Converting RGB image to grayscale and counting non-zero pixels.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.countNonZero(gray)
