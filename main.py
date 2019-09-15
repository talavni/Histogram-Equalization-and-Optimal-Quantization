import numpy as np
from skimage.color import rgb2yiq, yiq2rgb

def histogram_equalize(im_orig):
    """
    performs histogram equalization to the image
    :param im_orig: an ndimage array
    :return: array of equalized image, the original histogram and the cumulative histogram
    """
    if len(im_orig.shape) > 2:
        # rgb
        YIQim = rgb2yiq(im_orig) * 255
        hist_orig, bin_edges = np.histogram(YIQim[:, :, 0], 256)
        rows, columns, dim = im_orig.shape
        cum_hist = np.cumsum(hist_orig)
        cum_hist = cum_hist.astype(np.float64)
    else:
        # grayscale
        im_orig *= 255
        hist_orig, bin_edges = np.histogram(im_orig, 256)
        cum_hist = np.cumsum(hist_orig)
        cum_hist = cum_hist.astype(np.float64)
        rows, columns = im_orig.shape

    tot_pixels = rows * columns
    cum_hist = (cum_hist/tot_pixels)
    minimum = min(np.nonzero(cum_hist)[0])
    maximum = np.nonzero(cum_hist)[0][-1]
    minVal = cum_hist[minimum]
    maxVal = cum_hist[maximum]
    cum_hist = (255 * ((cum_hist-minVal)/(maxVal-minVal)))
    cum_hist = np.around(cum_hist)
    if len(im_orig.shape) > 2:
        im_eq = np.copy(YIQim)
        y_values = cum_hist[YIQim[:, :, 0].astype(np.int8)]
        im_eq[:, :, 0] = y_values
        im_eq = yiq2rgb(im_eq/255)
    else:
        im_eq = cum_hist[im_orig.astype(np.int8)]
    cum_hist /= 255
    cum_hist = np.clip(cum_hist, 0, 1)
    return [im_eq, hist_orig, cum_hist]


def quantize(im_orig, n_quant, n_iter):
    """
    a function that performs optimal quantization of a given grayscale or RGB image.

    :param im_orig: is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: is the number of intensities your output im_quant image should have.
    :param n_iter: is the maximum number of iterations of the optimization procedure
    :return: im_quant - is the quantized output image.
            error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
            quantization procedure
    """
    error = []
    q = np.array([0] * n_quant, dtype=np.float64)
    z = [0] * (n_quant + 1)
    if len(im_orig.shape) > 2:
        # rgb
        YIQim = rgb2yiq(im_orig)
        hist = np.histogram(YIQim[:, :, 0] * 255, bins=256)[0]
    else:
        # grayscale
        hist = np.histogram(im_orig * 255, bins=256)[0]
    cum_hist = np.cumsum(hist)
    for i in range(1, n_quant):
        z[i] = np.where(cum_hist > (i/n_quant) * cum_hist[255])[0][0] - 1
    z[n_quant] = 255

    q_1 = 0
    q_2 = 0
    for i in range(n_iter):
        change = False  # boolean to see if z changed

        # calculate new q
        for j in range(n_quant):
            for x in range(z[j], z[j+1] + 1):
                q_2 += hist[x]
                q_1 += (x * hist[x])
            q[j] = int(q_1/q_2)
            q_1 = 0
            q_2 = 0

            # calculate new z
        for j in range(1, n_quant):
            z_1 = int((q[j-1] + q[j]) / 2)
            if z_1 != z[j]:
                z[j] = z_1
                change = True

        error.append(compute_error(n_quant, z, q, hist))
        if not change:
            lut = create_lut(z, q)
            if len(im_orig.shape) > 2:
                YIQim *= 255
                im_quant = np.copy(YIQim)
                y_values = lut[YIQim[:, :, 0].astype(np.int8)]
                im_quant[:, :, 0] = y_values
                im_quant = yiq2rgb(im_quant)
            else:
                im_orig *= 255
                im_quant = lut[im_orig.astype(np.int8)]
            return [im_quant, error]

    lut = create_lut(z, q)
    if len(im_orig.shape) > 2:
        YIQim *= 255
        im_quant = np.copy(YIQim)
        y_values = lut[YIQim[:, :, 0].astype(np.int8)]
        im_quant[:, :, 0] = y_values
        im_quant = yiq2rgb(im_quant / 255)
    else:
        im_orig *= 255
        im_quant = lut[im_orig.astype(np.int8)]
    return [im_quant, error]


def create_lut(z, q):
    """
    create lookup table
    """
    lut = []
    for i in range(1, len(z)):
        lut = np.concatenate((lut, [int(q[i-1])] * (z[i]-z[i-1])), axis=None)
    lut = np.concatenate((lut, [z[-1]]), axis=None)
    return lut


def compute_error(n_quant, z, q, hist):
    sum = 0
    for i in range(n_quant):
        for x in range(z[i], z[i + 1] + 1):
            sum += ((q[i] - x) ** 2) * hist[x]
    return sum
