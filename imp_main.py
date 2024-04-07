import argparse
from cv2 import imread, imwrite
import filters

filters_table = {
    'crop': filters.crop,
    'gs': filters.grayscale,
    'neg': filters.negative,
    'sharp': filters.sharpening,
    'edge': filters.edge_detection,
    'blur': filters.gaussian_blur,
    'cryst': filters.crystallize
}

def parse(values, type: type = None, name = '<error-name>', args = []):
    if len(values) == len(args):
        try:
            return [type(v) for v in values]
        except ValueError:
            raise ValueError('-{0} argument(s) must be {1}'.format(name, str(type).split()[1].removesuffix('>')))
    if len(args) > 0:
        raise ValueError('-{0} takes {1} integer argument(s): {2}'.format(name, len(args), ', '.join(args)))
    raise ValueError("-{0} doesn't take arguments".format(name))
    
parse_kwargs = {
    'crop': {'type': int, 'name': 'crop', 'args': ['width', 'height']},
    'gs': {'name': 'gs'},
    'neg': {'name': 'neg'},
    'sharp': {'name': 'sharp'},
    'edge': {'type': int, 'name': 'edge', 'args': ['threshold']},
    'blur': {'type': float, 'name': 'blur', 'args': ['sigma']},
    'cryst': {'type': int, 'name': 'cryst', 'args': ['count']},
}

class PathAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != 2:
            raise ValueError("required to set input and output paths by 1st and 2nd arguments")
        setattr(namespace, 'image', imread(values[0]))
        setattr(namespace, 'output_path', values[1])
        if namespace.image is None:
            raise FileNotFoundError('incorrect input path')

class FilterAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, 'image'):
            raise ValueError('trying to process filter before input path specified')
        print('Processing -{0}...'.format(self.dest))
        namespace.image = filters_table[self.dest](namespace.image, *parse(values, **parse_kwargs[self.dest]))

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('input', nargs='*', action=PathAction)
for name in filters_table.keys():
    arg_parser.add_argument('-' + name, nargs='*', action=FilterAction)
    
namespace = arg_parser.parse_args()
imwrite(namespace.output_path, namespace.image)
