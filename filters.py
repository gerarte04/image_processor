import numpy as np
from tqdm.auto import tqdm

def split(image):
    image = np.swapaxes(image, 2, 0)
    return [image[i].transpose() for i in range(3)]

def apply_matrix(image, matrix):
    matrix = np.array(matrix)
    layers = [np.pad(l, 1, 'edge') for l in split(image.astype(np.int32))]
    for i in tqdm(range(1, image.shape[0] + 1)):
        for j in range(1, image.shape[1] + 1):
            image[i-1,j-1] = [np.sum(np.multiply(l[i-1:i+2, j-1:j+2], matrix)) for l in layers]

    return image.astype(np.uint8)

def crop(image, width=None, height=None):
    return image[0:height, 0:width]

def grayscale(image):
    layers = split(image)
    image = 0.299 * layers[0] + 0.587 * layers[1] + 0.114 * layers[2]
    np.astype(image, )
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

def edge_detection(image, threshold):
    image = apply_matrix(grayscale(image),
        [[0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]]
    )
    return np.select([image > threshold], [255], 0)

def gaussian_blur(image, sigma):
    pass
