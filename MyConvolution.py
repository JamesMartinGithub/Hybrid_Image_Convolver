import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray
    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    imgtype = image.dtype
    newimage = np.zeros(image.shape, dtype=imgtype)

    # 1) Flip kernel
    kernel = np.fliplr(np.flipud(kernel))

    height = image.shape[0]
    width = image.shape[1]
    kheight = kernel.shape[0]
    kwidth = kernel.shape[1]
    vpadding = kernel.shape[1] // 2  # For vertical axis
    hpadding = kernel.shape[0] // 2  # For horizontal axis
    # Convolution for greyscale images is separate to avoid unnecessarily tripling of computation
    if image.shape.__len__() == 2:
        # Image is greyscale
        # 2) Zero pad image
        if vpadding > 0:
            image = np.append(np.insert(image, np.zeros(vpadding, dtype=int), 0, axis=1), np.zeros((height, vpadding), dtype=int), axis=1)  # Vertical axis
        if hpadding > 0:
            image = np.append(np.insert(image, np.zeros(hpadding, dtype=int), 0, axis=0), np.zeros((hpadding, width + (vpadding*2)), dtype=int), axis=0)  # Horizontal axis

        # 3) Convolve
        for y in range(0, (image.shape[0]-(hpadding*2))):
            for x in range(0, (image.shape[1]-(vpadding*2))):
                newimage[y, x] = np.sum(np.multiply(kernel, image[y:(y+kheight), x:(x+kwidth)]))
    else:
        # Image is full colour
        # 2) zero pad image
        image = np.append(np.insert(image, np.zeros(vpadding, dtype=int), 0, axis=1), np.zeros((height,vpadding,3), dtype=int), axis=1)  # Vertical axis
        image = np.append(np.insert(image, np.zeros(hpadding, dtype=int), 0, axis=0), np.zeros((hpadding, width+(vpadding*2), 3), dtype=int), axis=0)  # Horizontal axis

        # 2.5) Increase kernel for 3 channel multiplication
        newkernel = np.zeros((kernel.shape[0], kernel.shape[1], 3))
        for yk in range(0, (kheight-1)):
            for xk in range(0, (kwidth-1)):
                newkernel[yk, xk] = np.repeat(kernel[yk, xk], 3)
        kernel = newkernel

        # 3) Convolve
        for y in range(0, (image.shape[0]-(hpadding*2))):
            for x in range(0, (image.shape[1]-(vpadding*2))):
                newimage[y, x] = np.sum(np.multiply(kernel, image[y:(y + kheight), x:(x + kwidth)]), (0, 1))

    #return newimage.astype(dtype=imgtype)
    return newimage
