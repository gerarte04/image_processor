import math
import numpy as np

def split(image):
    image = np.swapaxes(image, 2, 0)
    return [image[i].transpose() for i in range(3)]

def apply_matrix(image, matrix):
    n = np.size(matrix, axis=0)
    image = image.astype(np.int32)
    matrix = np.array(matrix).reshape(n*n, 1)
    layers = [np.pad(l, n // 2, 'edge') for l in split(image)]
    for i in range(image.shape[0]):
        new = None
        for l in layers:
            row = l[i:i+n]
            strips = np.concatenate([row[...,j:image.shape[1]+j] for j in range(n)], axis=0)
            new_l = np.sum(strips * matrix, axis=0)
            new_l = np.select([new_l > 255, new_l < 0], [255, 0], new_l)
            new_l = np.expand_dims(new_l, 1)
            if new is None:
                new = new_l
            else:
                new = np.concatenate((new, new_l), axis=1)
        image[i] = new

    return image.astype(np.uint8)

def crop(image, width=None, height=None):
    return image[0:height, 0:width]

def grayscale(image):
    layers = split(image)
    image = 0.299 * layers[0] + 0.587 * layers[1] + 0.114 * layers[2]
    image = np.expand_dims(image, 2)
    return np.pad(image.astype(np.uint8), ((0,0),(0,0),(0,2)), 'edge')

def negative(image):
    neg = lambda i: 255 - i
    return neg(image)

def sharpening(image):
    return apply_matrix(image,
        [[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]]
    )

def edge_detection(image, threshold=70):
    image = apply_matrix(grayscale(image),
        [[0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]]
    )
    return np.select([image > threshold], [255], 0)

def gaussian_blur(image, sigma=100):  # *= 100%
    sigma /= 100
    n = math.ceil(6*sigma)
    if n % 2 == 0:
        n += 1
    r = n // 2

    coeff_mtx = np.array([])
    for x in range(-r, r+1):
        for y in range(-r, r+1):
            coeff_mtx = np.append(coeff_mtx, x**2+y**2)
    coeff_mtx = np.exp(-coeff_mtx/(2*sigma**2)) / (2*math.pi*sigma**2)
    coeff_mtx = coeff_mtx.reshape(n, n)

    return apply_matrix(image, coeff_mtx)
