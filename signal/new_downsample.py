
import cv2
import numpy as np


def downsample(img, factor=2):
    downsampled_img = cv2.pyrDown(img)
    for i in range(factor-1):
        downsampled_img = cv2.pyrDown(downsampled_img)
    return downsampled_img




def fft1d(signal):
    N = len(signal)
    if N <= 1:
        return signal
    
def fft2d(image):

    image = np.atleast_2d(image)

    shape = image.shape
    padded_shape = tuple(2 ** int(np.ceil(np.log2(s))) for s in shape)
    pad_width = tuple((0, p - s) for s, p in zip(shape, padded_shape))
    padded_image = np.pad(image, pad_width, 'constant', constant_values=(0, 0))

    fft_image = np.fft.fft2(padded_image)
    return fft_image



def ifft1d(signal):
    N = len(signal)
    if N <= 1:
        return signal
    else:
 
        even = ifft1d(signal[::2])
        odd = ifft1d(signal[1::2])

        twiddles = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([even + twiddles[:N//2] * odd, even + twiddles[N//2:] * odd])
    
def ifft2d(fft_image):

    image = np.fft.ifft2(fft_image)
    # Remove any padding that was added to the image.
    unpadded_shape = tuple(s // 1 for s in fft_image.shape)
    unpadded_image = image[:unpadded_shape[0], :unpadded_shape[1]]

    return np.real(unpadded_image)


def fftshift(f):
 
    fshift = np.zeros_like(f, dtype=np.complex128)
    rows, cols = len(downsampled_img), len(downsampled_img[0])
    crow, ccol = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            fshift[i, j] = f[(i + crow) % rows, (j + ccol) % cols]
    return fshift




def ifftshift(x):
    x = np.asarray(x)
    shift = [dim // 2 for dim in x.shape]
    for i in range(x.ndim):
        x = np.roll(x, shift[i], axis=i)
    return x




if __name__ == '__main__':
    # Load the image
    img = cv2.imread('C:\\Users\\bdbir\\Desktop\\signal\\sprite.png')
    downsampled_img=downsample(img)

    f = fft2d(downsampled_img)

    fshift =fftshift(f)

    rows, cols = downsampled_img.shape[:2]
    crow, ccol = rows//2, cols//2
    r = 120
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), r, 255, -1)

    resized_mask = cv2.resize(mask, (cols, rows))

    resized_mask = cv2.merge([resized_mask, resized_mask, resized_mask])

    resized_mask = cv2.resize(mask, (cols, rows))

    f_ishift =ifftshift(fshift)


    img_back =ifft2d(f_ishift)

    img_back = np.abs(img_back)

    img_back = np.uint8(img_back)

    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('Downsampled Image', img_back)
    cv2.waitKey(0)
    cv2.destroyAllWindows()