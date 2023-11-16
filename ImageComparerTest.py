import imageio.v3 as iio
import numpy as np
def runTest():
    imgfile = "NewHybridTest.bmp"
    img1 = iio.imread(imgfile)
    imgfile = "NewHybridTest2.bmp"
    img2 = iio.imread(imgfile)

    img3 = np.zeros(img1.shape, dtype=np.uint8)
    for h in range(0, img1.shape[0]):
        for w in range(0, img1.shape[1]):
            if np.array_equal(img1[h, w], img2[h, w]):
                img3[h, w] = img1[h, w]
            else:
                img3[h, w] = 0

    iio.imwrite("ComparisonResult.bmp", img3)
    print("Result saved to ComparisonResult.bmp")

if __name__ == '__main__':
    runTest()