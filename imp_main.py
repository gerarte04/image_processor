import argparse
from cv2 import imread, imwrite
import filters

filters_table = {
    'crop': filters.crop,
    'gs': filters.grayscale,
    'neg': filters.negative,
    'sharp': filters.sharpening,
    'edge': filters.edge_detection,
    'blur': filters.gaussian_blur
}

class PathAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != 2:
            raise ValueError("required to set input and output paths by first 2 arguments")
        setattr(namespace, 'image', imread(values[0]))
        setattr(namespace, 'output_path', values[1])

class FilterAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.image = filters_table[self.dest](namespace.image, *[int(v) for v in values])

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('input_path', nargs=2, action=PathAction)
for name in filters_table.keys():
    arg_parser.add_argument('-' + name, nargs='*', action=FilterAction)
    
namespace = arg_parser.parse_args()
imwrite(namespace.output_path, namespace.image)
