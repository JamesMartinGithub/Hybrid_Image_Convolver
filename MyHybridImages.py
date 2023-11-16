import math
import numpy as np

from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma:
float) -> np.ndarray:
    """ 
    Create hybrid images by combining a low-pass and high-pass filtered pair. 
    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or 
    colour shape=(rows,cols,channels)) 
    :type numpy.ndarray 
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering 
    lowImage 
    :type float 
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or 
    colour shape=(rows,cols,channels)) 
    :type numpy.ndarray 
    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering 
    highImage before subtraction to create the high-pass filtered image 
    :type float  
    :returns returns the hybrid image created 
    by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it 
    with  
    a high-pass image created by subtracting highImage from highImage convolved with 
    a Gaussian of s.d. highSigma. The resultant image has the same size as the input 
    images. 
    :rtype numpy.ndarray 
    """

    # Convert image types to signed int to allow negatives
    #lowImage = lowImage.astype(dtype=np.int16)
    #highImage = highImage.astype(dtype=np.int16)

    # If the images are greyscale and full colour, convert the greyscale to full colour
    if lowImage.shape.__len__() != highImage.shape.__len__():
        if lowImage.shape.__len__() == 2:
            lowImage = convertGreyscaleImage(lowImage)
        else:
            highImage = convertGreyscaleImage(highImage)

    # Convolve images with gaussian kernel
    processedlowimg = convolve(lowImage, makeGaussianKernel(lowSigma))
    processedhighimg = highImage - convolve(highImage, makeGaussianKernel(highSigma))

    # Sum high and low passes
    result = processedlowimg + processedhighimg

    # Clamp all values between 0-255 to avoid artifacts
    result = np.clip(result, a_min=0, a_max=255)

    # Convert image types back to unsigned int
    #result = result.astype(dtype=np.uint8)

    return result


def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Returns a 2D gaussian kernel with standard deviation sigma.
    """

    # Calculate size from sigma
    size = math.floor(8 * sigma + 1)
    if size % 2 == 0:
        size += 1

    # Create kernel
    kernel = np.zeros((size,size))
    edge = size // 2
    for y in range(0, size):
        for x in range(0, size):
            kernel[y, x] = getGaussianValue(x - edge, y - edge, sigma)

    # Rescale kernel values to ensure kernel sums to 1
    scalar = ((1 - np.sum(kernel)) + 1)
    if scalar != 1:
        for y in range(0, size):
            for x in range(0, size):
                kernel[y, x] = kernel[y, x] * scalar

    return kernel


def getGaussianValue(x: int, y: int, sigma: float) -> float:
    """
    Returns the gaussian value at (x,y) given sigma
    """

    result = (1.0/(2.0*math.pi*(math.pow(sigma, 2)))) * math.pow(math.e, ((-((math.pow(x, 2))+(math.pow(y, 2))))/(2*(math.pow(sigma, 2)))))

    return result


def convertGreyscaleImage(image: np.ndarray) -> np.ndarray:
    """
    Converts a 2D greyscale image to a 3D full-colour image by repeating each colour value 3 times
    """

    newimage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.int16)
    for y in range(0, (image.shape[0] - 1)):
        for x in range(0, (image.shape[1] - 1)):
            newimage[y, x] = np.repeat(image[y, x], 3)

    #return newimage.astype(dtype=np.int16)
    return newimage
