import glob
import cv2
import numpy as np

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from utils.swirHeader import swirHeader

def openSwir(file, swirhd):
    file_handle = open(file, 'rb')
    swirimg = np.fromfile(file_handle, dtype=np.uint16, count=swirhd.lines * swirhd.bands * swirhd.samples)
    swirimg = swirimg.reshape(swirhd.lines, swirhd.bands, swirhd.samples)

    file_handle.close()
    return swirimg

def denoiseLinedefectSwir(swirimg, swirhd):
    linedefect = swirhd.linedefect.astype(np.uint16)

    for i, x in linedefect:
        swirimg[:, i, x] = (swirimg[:, i, x - 1] + swirimg[:, i, x + 1]) / 2

    return swirimg


def getRectROI(swirimg, swirhd):
    bandForThreshold = 178
    treshold_val = 127
    swirimg_array = swirimg[:, bandForThreshold, :] / (1 << (swirhd.datatype - 8))
    swirimg_array = np.clip(swirimg_array, 0, 255)
    img = swirimg_array.astype(np.uint8)

    img = cv2.threshold(img, treshold_val, 255, cv2.THRESH_BINARY)[1]

    ret, labels = cv2.connectedComponents(img)


def select_largest_obj(img_bin, lab_val=255, fill_holes=False, smooth_boundary=False, kernel_size=15):
    n_labels, img_labeled, lab_stats, _ = \
        cv2.connectedComponentsWithStats(img_bin, connectivity=8,
                                         ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val

    lblareas = lab_stats[1:, cv2.CC_STAT_AREA]
    imax = max(enumerate(lblareas), key=(lambda x: x[1]))[0] + 1

    boundrect = np.zeros(shape=(4))
    boundrect[0] = lab_stats[imax, cv2.CC_STAT_LEFT]
    boundrect[1] = lab_stats[imax, cv2.CC_STAT_TOP]
    boundrect[2] = lab_stats[imax, cv2.CC_STAT_WIDTH]
    boundrect[3] = lab_stats[imax, cv2.CC_STAT_HEIGHT]

    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed,
                      newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)

    return boundrect, largest_mask


def getSwirMask(swirimg, labelbandnum, bitcount, threshold):
    swirimg_array = swirimg[:, labelbandnum, :] / (1 << (bitcount - 8))
    swirimg_array = np.clip(swirimg_array, 0, 255)
    new_image = swirimg_array.astype(np.uint8)
    ret, img_binary = cv2.threshold(new_image, threshold, 255, cv2.THRESH_BINARY_INV)
    boundrect, mask = select_largest_obj(img_binary, lab_val=255, fill_holes=True, smooth_boundary=True, kernel_size=5)

    # cv2.imshow("showimgSwir", mask)
    # cv2.waitKey(5)

    return boundrect, mask


def getCropRect(boundrect, cropw, croph):
    #     centerx = boundrect[0] + boundrect[2]/2
    #     centery = boundrect[1] + boundrect[3]/2
    #     croprect = np.zeros(shape = (4))
    #
    #     croprect[0] = centerx - cropw/2
    #     croprect[1] = centery - croph/2
    #     croprect[2] = cropw
    #     croprect[3] = croph

    croprect = boundrect

    return croprect


def getSwirWithMaskcrop(swirimg, bandCount, bitcount, mask, croprect, resizeshape):
    swirdata = np.array([]).astype(np.float32)
    for i in range(bandCount):
        swirimg_array = swirimg[:, i, :] / (1 << (bitcount - 8))
        swirimg_array = np.clip(swirimg_array, 0, 255)
        new_image = swirimg_array.astype(np.uint8)
        new_image = cv2.bitwise_and(new_image, mask)

        y1 = croprect[1].astype(np.int16)
        if (y1 < 0):
            y1 = 0
        y2 = (croprect[1] + croprect[3]).astype(np.int16)

        x1 = croprect[0].astype(np.int16)
        if (x1 < 0):
            x1 = 0
        x2 = (croprect[0] + croprect[2]).astype(np.int16)

        crop_img = new_image[y1:y2, x1:x2]

        rzimg = cv2.resize(crop_img, dsize=(resizeshape, resizeshape), interpolation=cv2.INTER_LINEAR)

        fimg = rzimg.astype(np.float32)
        fimg = preprocess_input(fimg)
        fimg1d = fimg.ravel().astype(np.float32)
        swirdata = np.append(swirdata, fimg1d)

    return swirdata


def getRGBapplenpycode(rawfile):
    pathnpy = rawfile.rsplit('/', maxsplit=1)[0]
    npyfiles = [f for f in glob.glob(pathnpy + "**/*.npy", recursive=True)]
    num = 0

    for npyfile in npyfiles:
        filename = npyfile.rsplit('/', maxsplit=1)[1].replace('.npy', '')
        num = float(filename)

    return num


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


def preprocess_input2(x):
    x /= 255.
    return x

def getStridedMat2d(mat2d, stridesize):
    stridei = np.array(list(range(0, np.shape(mat2d)[0], stridesize)))
    StridedMat2d = mat2d[:, stridei]
    StridedMat2d = StridedMat2d[stridei, :]

    return StridedMat2d


def loadswirimage(path, datasize):
    x = np.load(path)
    x = x[:datasize * datasize * datasize]
    x = x.reshape((datasize, datasize, datasize))
    # x = np.transpose(x, (0, 2, 1))
    x = np.transpose(x, (1, 2, 0))
    return x

def preprocess_swir(rawfile, resizeshape):

    #cropw = 194
    cropw = 194
    croph = 352
    datasize = 256
    bandstride = int(datasize / resizeshape)

    swirhd = swirHeader()
    swirimg = openSwir(rawfile, swirhd)
    swirimg = denoiseLinedefectSwir(swirimg, swirhd)

    boundrect, mask = getSwirMask(swirimg, swirhd.labelbandnum, swirhd.datatype, threshold=60)

    # print('w:{0} h:{1}'.format(boundrect[2], boundrect[3]))
    croprect = getCropRect(boundrect, cropw, croph)
    swirmaskcrop = getSwirWithMaskcrop(swirimg, swirhd.bands, swirhd.datatype, mask, croprect, datasize)
    swirmaskcrop = swirmaskcrop.reshape((datasize * datasize * swirhd.bands))

    if resizeshape == 256:
        return swirmaskcrop

    swir3d = swirmaskcrop.reshape(datasize, datasize, datasize)
    resizedswir = np.array([]).astype(np.float32)

    for i in range(0, datasize, bandstride):
        swir2d = swir3d[i, :, :]
        swirstride2d = getStridedMat2d(swir2d, bandstride)
        resizedswir = np.append(resizedswir, swirstride2d)

    resizedswir = resizedswir.reshape((resizeshape * resizeshape * resizeshape))

    return resizedswir

if __name__ == '__main__':
    # import volumerender_sim as volr
    # rawfile = '/hdd/_data/_Food/tomato/201804_ripeness/0/0001.raw'
    rawfile = 'F:\\_data\\_Food\\_tomato\\201804_데프니스_LR저장\\2018-04-13\\041301.raw'
    #rawfile = 'F:/_data/_Food/_tomato/201804_ripeness/0/0001.raw'
    pswir = preprocess_swir(rawfile, 256)
    print(f"type pswir:{type(pswir)}")
    print(f"numpy shape:{np.shape(pswir)}")

    # volr.volumerender_sim(pswir, 256, 256, 256)


    ##for j in range(256):
        ##swirimg = pswir[j, :, :]

        ##print(np.shape(swirimg))

        ##cv2.imshow("showimgSwir", swirimg)
        ##cv2.waitKey(0)




