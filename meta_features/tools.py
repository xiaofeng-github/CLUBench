__author__ = 'XF'
__date__ = '2024/08/30'

'The Script aims to set frequently used functional tools.'

# built-in library
import os
import os.path as osp
import time
import pickle


def generate_filename(suffix, *args, sep='_', timestamp=False):

    '''

    :param suffix: suffix of file
    :param sep: separator，default '_'
    :param timestamp: add timestamp for uniqueness
    :param args:
    :return:
    '''

    filename = sep.join(args).replace(' ', '_')
    if timestamp:
        filename += time.strftime('_%Y%m%d%H%M%S')
    if suffix[0] == '.':
        filename += suffix
    else:
        filename += ('.' + suffix)

    return filename
    

# object serialization
def obj_save(path, obj):

    if obj is not None:
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    else:
        print('object is None!')


# object instantiation
def obj_load(path):

    if osp.exists(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    else:
        raise OSError('no such path:%s' % path)


def new_dir(father_dir, mk_dir=None):

    if mk_dir is not None:
        new_path = osp.join(father_dir, mk_dir)
    else:   
        new_path = osp.join(father_dir, time.strftime('%Y%m%d%H%M%S'))
    if not osp.exists(new_path):
        os.makedirs(new_path)
    return new_path
