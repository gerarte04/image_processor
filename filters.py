import numpy as np

def split(image):
    image = np.swapaxes(image, 2, 0)
    return [image[i].transpose() for i in range(3)]

def apply_matrix(image, matrix):
    image = image.astype(np.int32)
    matrix = np.array(matrix).reshape(9, 1)
    layers = [np.pad(l, 1, 'edge') for l in split(image)]
    for i in range(0, image.shape[0]):
        new = None
        for l in layers:
            row = l[i:i+3]
            strips = np.concatenate((row[...,0:image.shape[1]],
                                    row[...,1:image.shape[1]+1],
                                    row[...,2:image.shape[1]+2]), axis=0)
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

def gaussian_blur(image, sigma):
    pass
