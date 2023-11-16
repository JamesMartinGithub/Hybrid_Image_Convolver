import imageio.v3 as iio
import numpy as np
import MyConvolution as convolution
import MyHybridImages as hybrid


def testConvolution():
    # image import
    imgfile = input("Image URI: ")
    image = iio.imread(imgfile)

    """
    # grayscale array creation
    gimage = np.array([[0, 0, 64, 64, 255, 255],
                        [0, 0, 64, 64, 255, 255],
                        [64, 64, 64, 64, 255, 255],
                        [64, 64, 64, 64, 255, 255],
                        [255, 255, 255, 255, 255, 255],
                        [255, 255, 255, 255, 255, 255]])

    # use greyscale for test
    #image = gimage
    """

    # create 3x3 identity kernel
    #kernel = np.array([[0, 0, 0],
    #                   [0, 1, 0],
    #                   [0, 0, 0]])
    # create 3x5 identity kernel
    #kernel = np.array([[0, 0, 0],
    #                   [0, 0, 0],
    #                   [0, 1, 0],
    #                   [0, 0, 0],
    #                   [0, 0, 0]])
    # create 5x3 identity kernel
    #kernel = np.array([[0, 0, 0, 0, 0],
    #                   [0, 0, 1, 0, 0],
    #                   [0, 0, 0, 0, 0]])
    # create 3x3 blur kernel
    kernel = np.repeat(1/63, 63)
    kernel = kernel.reshape((7,9))
    print(kernel)

    # Convolve
    result = convolution.convolve(image, kernel)

    # Convert for printing
    np.printoptions()
    for i in image:
        print(np.array_repr(i).replace('\n', ''))

    for i in result:
        print(np.array_repr(i).replace('\n', ''))

    # Save result to file
    iio.imwrite("resultimg.bmp", result)


def testHybrid():

    # Image imports
    imgfile = input("Lowpass Image URI: ")
    #imgfile = "snowboarder.bmp"
    lowimage = iio.imread(imgfile)
    imgfile = input("Highpass Image URI: ")
    #imgfile = "skater.bmp"
    highimage = iio.imread(imgfile)
    sigma1 = int(input("Sigma1 value: "))
    #sigma1 = 2
    sigma2 = int(input("Sigma2 value: "))
    #sigma2 = 6

    # Compute
    result = hybrid.myHybridImages(lowimage, sigma1, highimage, sigma2)

    # Save result to file
    iio.imwrite("NewHybridTest2.bmp", result)
    print("Result saved to NewHybridTest2.bmp")

    # Convert for printing
    #with np.printoptions(threshold=sys.maxsize):
    #    for i in result:
    #        print(np.array_repr(i).replace('\n', ''))



if __name__ == '__main__':
    #testConvolution()
    testHybrid()
