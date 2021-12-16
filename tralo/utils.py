import numpy as np
import os
import re
import torch
import math
import gzip
import json
from tralo.log import log

from functools import partial
from inspect import signature, getsource

from shutil import copy, copytree

from os.path import join, dirname, realpath, expanduser, isfile, islink, relpath, isdir, basename



class AttributeDict(dict):
    """ 
    An extended dictionary that allows access to elements as atttributes and counts 
    these accesses. This way, we know if some attributes were never used. 
    """

    def __init__(self, *args, **kwargs):
        from collections import Counter
        super().__init__(*args, **kwargs)
        self.__dict__['counter'] = Counter()

    def __getitem__(self, k):
        self.__dict__['counter'][k] += 1
        return super().__getitem__(k)

    def __getattr__(self, k):
        self.__dict__['counter'][k] += 1
        return super().get(k)

    def __setattr__(self, k, v):
        return super().__setitem__(k, v)

    def __delattr__(self, k, v):
        return super().__delitem__(k, v)    

    def unused_keys(self, exceptions=()):
        return [k for k in super().keys() if self.__dict__['counter'][k] == 0 and k not in exceptions]

    def assume_no_unused_keys(self, exceptions=()):
        if len(self.unused_keys(exceptions=exceptions)) > 0:
            log.warning('Unused keys:', self.unused_keys(exceptions=exceptions))


def get_attribute(name):
    import importlib

    if name is None:
        raise ValueError('The provided attribute is None')
    
    name_split = name.split('.')
    mod = importlib.import_module('.'.join(name_split[:-1]))
    return getattr(mod, name_split[-1])


def count_parameters(model, only_trainable=False):
    """ Count the number of parameters of a torch model. """
    import numpy as np
    return sum([np.prod(p.size()) for p in model.parameters()
                if (only_trainable and p.requires_grad) or not only_trainable])


def dict_to_str(p):
    if type(p) == bool:
        return 'y' if p else 'n'
    elif type(p) in {list, tuple}:
        return '_'.join(dict_to_str(a) for a in p)
    elif type(p) == dict:
        return '_'.join(k + '-' + dict_to_str(p[k]) for k in sorted(p.keys()))
    elif p is None:
        return 'N'
    else:
        return str(p)

def get_current_git(path=None):
    from subprocess import run, PIPE

    if path is None:
        path = join(dirname(realpath(__file__)), '..', '..',)
    else:
        path = dirname(path)

    out = run(["git", "rev-parse", "HEAD"], stdout=PIPE, stderr=PIPE, cwd=path)

    if out.returncode == 0:
        return out.stdout.decode('utf8')[:-1]
    else:
        return None


def str_recursive(obj):
    if isinstance(obj, (tuple, list)):
        return '_'.join(str_recursive(x) for x in obj)

    if isinstance(obj, (dict,)):
        return '_'.join((str(k) + '-' + str_recursive(obj[k])) for k in sorted(obj.keys()))

    if isinstance(obj, (set, frozenset)):
        return '_'.join(sorted(str_recursive(x) for x in obj))

    return str(obj)


# convenience

def get_batch(dataset, n_or_indices_or_start, end=None, cuda=False, shuffle=False):
    samples = []

    # process input arguments
    indices = None
    if type(n_or_indices_or_start) == int and end is None:
        start = 0
        end = n_or_indices_or_start
    elif type(n_or_indices_or_start) == int and end is not None:
        start = n_or_indices_or_start
    elif type(n_or_indices_or_start) in {list, tuple}:
        indices = n_or_indices_or_start

    if indices is None:
        if not shuffle:
            indices = range(start, end)
        else:
            import random
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[start: end]

    for i in indices:
        samples += [dataset[i]]

    assert len(set(([(len(sx), len(sy)) for sx, sy in samples]))) == 1

    from torch.utils.data._utils.collate import default_collate

    batch_x, batch_y = default_collate(samples)

    if cuda:
        batch_x = [x.cuda() if type(x) == torch.Tensor else x for x in batch_x]
        batch_y = [y.cuda() if type(y) == torch.Tensor else y for y in batch_y]

    return batch_x, batch_y


# function signature

def filter_args(input_args, default_args):

    updated_args = {k: input_args[k] if k in input_args else v for k, v in default_args.items()}
    used_args = {k: v for k, v in input_args.items() if k in default_args}
    unused_args = {k: v for k, v in input_args.items() if k not in default_args}

    return AttributeDict(updated_args), AttributeDict(used_args), AttributeDict(unused_args)


# folders and find

def valid_run(folder_name):
    return all([
        isfile(realpath(join(folder_name, f))) 
        for f in ['stats.json', 'config.json']
    ]) and any([
        f.startswith('weights') and f.endswith('.pth') 
        for f in os.listdir(realpath(join(folder_name)))
    ])

def find_in_folders(substr, base_folders, must_end_with=None):
    """ searches base_folders recursively for files that match substr """

    matching = []

    if type(base_folders) not in {list, tuple}:
        base_folders = [base_folders]

    for folder in base_folders:
        matching += [join(m_path, m) for m_path, _, files in os.walk(expanduser(folder)) for m in files
                     if m.startswith(substr) and not re.match(r'^.*(\.log-?.*|-args)$', m)]

    if must_end_with is not None:
        matching = [m for m in matching if m.endswith(must_end_with)]

    return matching

# TODO: integrity check per file
def get_from_repository(local_name, repo_files, integrity_check=None, repo_dir='~/dataset_repository', 
                        local_dir='~/datasets'):
    """ copies files from repository to local folder.
    
    repo_files: list of filenames or list of tuples [filename, target path] 

    e.g. get_from_repository('MyDataset', [['data/dataset1.tar', 'other/path/ds03.tar'])
    will create a folder 'MyDataset' in local_dir, and extract the content of
    '<repo_dir>/data/dataset1.tar' to <local_dir>/MyDataset/other/path.
    """

    local_dir = realpath(join(expanduser(local_dir), local_name))

    dataset_exists = True

    # check if folder is available
    if not isdir(local_dir):
        dataset_exists = False

    if integrity_check is not None:
        try:
            integrity_ok = integrity_check(local_dir)
        except BaseException as e:
            print(e)
            integrity_ok = False

        if integrity_ok:
            repo_str = basename(repo_files[0][0]) + ('...' if len(repo_files) > 1 else '')
            log.hint(f'{repo_str}: Passed custom integrity check')
        else:
            log.hint('Custom integrity check failed')

        dataset_exists = dataset_exists and integrity_ok

    if not dataset_exists:

        repo_dir = realpath(expanduser(repo_dir))

        for i, filename in enumerate(repo_files):

            if type(filename) == str:
                origin, target = filename, filename
                archive_target = join(local_dir, basename(origin))
                extract_target = join(local_dir)
            else:
                origin, target = filename
                archive_target = join(local_dir, dirname(target), basename(origin))
                extract_target = join(local_dir, dirname(target))
            
            archive_origin = join(repo_dir, origin)

            log.hint(f'copy: {archive_origin} to {archive_target}')
            # make sure the path exists
            os.makedirs(dirname(archive_target), exist_ok=True)

            # file locking prevents multiple processes 
            from datetime import datetime
            import time 
            lock_file = archive_target + '.lock'
            max_lock_time = 60  # in seconds

            def file_blocked(file):
                if isfile(lock_file):
                    with open(lock_file) as fh:
                        return int(datetime.now().timestamp()) < int(fh.read()) + max_lock_time
                else:
                    return False

            while file_blocked(lock_file):
                time.sleep(0.2)
                print('file locked... waiting')

            with open(lock_file, 'w') as fh:
                fh.write(str(int(datetime.now().timestamp())))

            if os.path.isfile(archive_target):
                # only copy if size differs
                if os.path.getsize(archive_target) != os.path.getsize(archive_origin):
                    log.hint(f'file exists but filesize differs: target {os.path.getsize(archive_target)} vs. origin {os.path.getsize(archive_origin)}')
                    copy(archive_origin, archive_target)
                else:
                    log.hint('file sizes match, no copy necessary')
            else:
                log.hint('no archive file available, need to copy')
                copy(archive_origin, archive_target)

            extract_archive(archive_target, extract_target, noarchive_ok=True)

            if isfile(lock_file):
                os.unlink(lock_file)
            else:
                log.warning('lock file was deleted before extraction was done. Normally, this should not happen.')

            # concurrent processes might have deleted the file
            if os.path.isfile(archive_target):
                os.remove(archive_target)


# hashing

def sha1_hash_object(obj):
    import hashlib
    import base64

    obj_str = str(hash_object_recursive(obj)).encode('utf8')

    hash_str = base64.b64encode(hashlib.sha1(obj_str).digest())
    hash_str = hash_str.decode('utf-8').replace('/', '-').replace('+', '_')[:-1]
    return hash_str


def hash_object_recursive(obj):

    if isinstance(obj, (tuple, list, np.ndarray)):
        return tuple(hash_object_recursive(x) for x in obj)

    if isinstance(obj, (dict,)):
        return tuple(sorted((k, hash_object_recursive(obj[k])) for k in obj.keys()))

    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(hash_object_recursive(x) for x in obj))

    if type(obj) == type:
        return obj.__name__

    if callable(obj):
        if hasattr(obj, '__code__'):
            return hash_object_recursive(getsource(obj)), hash_object_recursive({p.name: p.default for p in signature(obj).parameters.values()})
        elif type(obj) == partial:
            return hash_object_recursive(obj.func), hash_object_recursive(obj.args), hash_object_recursive(obj.keywords)
        else:
            a = [getattr(obj, f) for f in dir(obj) if callable(getattr(obj, f))]
            return tuple(x.__code__.co_code for x in a if hasattr(x, '__code__'))

    return obj


# archive

def extract_archive(filename, target_folder=None, noarchive_ok=False):
    from subprocess import run, PIPE

    if filename.endswith('.tgz') or filename.endswith('.tar'):
        command = f'tar -xf {filename}'
        command += f' -C {target_folder}' if target_folder is not None else ''
    elif filename.endswith('.tar.gz'):
        command = f'tar -xzf {filename}'
        command += f' -C {target_folder}' if target_folder is not None else ''
    elif filename.endswith('zip'):
        command = f'unzip {filename}'
        command += f' -d {target_folder}' if target_folder is not None else ''
    else:
        if noarchive_ok:
            return
        else:
            raise ValueError(f'unsuppored file ending of {filename}')

    log.hint(command)
    result = run(command.split(), stdout=PIPE, stderr=PIPE)
    if result.returncode != 0:
        print(result.stdout, result.stderr)
    

def gzip_write(filename, data):
    with gzip.open(filename, 'wb') as fh:
        fh.write(json.dumps(data).encode('utf-8'))

def gzip_load(filename):
    with gzip.open(filename, 'rb') as fh:
        return fh.read()


# image
def resize(img, size, interpolation='bilinear', max_bound=False, min_bound=False, channel_dim=0):
    """ convenience wrapper of resize """
    from torchvision.transforms.functional import resize as torch_resize
    from torchvision.transforms import InterpolationMode

    assert channel_dim in {0, 2, None}
    assert not min_bound or not max_bound

    to_numpy, drop_first = False, False
    if type(img) == np.ndarray:
        img, to_numpy = torch.tensor(img), True

    if channel_dim == 2:
        img = img.permute(2, 0, 1)

    if len(img.shape) == 2:
        img, drop_first = img.unsqueeze(0), True        

    if min_bound or max_bound:
        factors = size[0] / img.shape[1], size[1] / img.shape[2] 
        if min_bound:
            i = 0 if factors[0] > factors[1] else 1
        else:
            i = 0 if factors[0] < factors[1] else 1
        target_size = [int(img.shape[1] * factors[i]), int(img.shape[2] * factors[i])]
        target_size[i] = size[i]
        size = target_size
    
    interpolations = {
        'nearest': InterpolationMode.NEAREST,
        'bilinear': InterpolationMode.BILINEAR,
        'bicubic': InterpolationMode.BICUBIC,
    }
    img_resized = torch_resize(img, size, interpolations[interpolation])

    if drop_first:
        img_resized = img_resized[0]

    if channel_dim == 2:
        img_resized = img_resized.permute(1,2, 0)

    if to_numpy:
        img_resized = img_resized.numpy()

    return img_resized


def images_to_grid(images, target_size, layout=None, spacing=5, scale_to_fit=True):


    if type(images) == np.ndarray:
        images = torch.from_numpy(images)

    if layout is None:
        d0 = math.ceil(math.sqrt(images.shape[0]))
        d1 = (d0 - 1) if d0 * (d0 - 1) >= images.shape[0] else d0
    else:
        d0, d1 = layout

    slice_max_s = min(target_size[0] // d0, target_size[1] // d1)
    slice_max_s = (int(slice_max_s), int(slice_max_s))

    if scale_to_fit:
        tf_data = [resize(images[s], slice_max_s, max_bound=True, channel_dim=2) for s in range(len(images))]
    else:
        tf_data = images

    slice_s = list(tf_data[0].shape)
    slice_s = [slice_s[0] + spacing, slice_s[1] + spacing]

    grid_image_size = (int(d0 * slice_s[0]) + 1, int(d1 * slice_s[1]) + 1) + ((3,) if len(images[0].shape) == 3 else ())
    grid_image = torch.ones(grid_image_size)

    for i in range(len(tf_data)):
        iy, ix = int(i // d1), int(i % d1)
        off_y, off_x = iy * slice_s[0], ix * slice_s[1]
        grid_image[off_y: off_y + tf_data[i].shape[0], off_x: off_x + tf_data[i].shape[1]] = tf_data[i]

    return grid_image


def random_crop_slices(origin_size, target_size):
    """Gets slices of a random crop. """
    assert origin_size[0] >= target_size[0] and origin_size[1] >= target_size[1], f'actual size: {origin_size}, target size: {target_size}'

    offset_y = torch.randint(0, origin_size[0] - target_size[0] + 1, (1,)).item()  # range: 0 <= value < high
    offset_x = torch.randint(0, origin_size[1] - target_size[1] + 1, (1,)).item()

    return slice(offset_y, offset_y + target_size[0]), slice(offset_x, offset_x + target_size[1])


def random_crop(tensor, target_size, spatial_dims=(1, 2)):
    """ Randomly samples a crop of size `target_size` from `tensor` along `image_dimensions` """
    assert len(spatial_dims) == 2 and type(spatial_dims[0]) == int and type(spatial_dims[1]) == int

    # slices = random_crop_slices(tensor, target_size, image_dimensions)
    origin_size = tensor.shape[spatial_dims[0]], tensor.shape[spatial_dims[1]]
    slices_y, slices_x = random_crop_slices(origin_size, target_size)

    slices = [slice(0, None) for _ in range(len(tensor.shape))]
    slices[spatial_dims[0]] = slices_y
    slices[spatial_dims[1]] = slices_x
    slices = tuple(slices)
    return tensor[slices]


def color_label_image(labels, n_colors=None):
    """ Create a color image from a label image consisting of category ids per pixel. """
    n_colors = n_colors if n_colors is not None else max(20, labels.max() + 1)

    from matplotlib import pyplot as plt

    colors = [plt.cm.turbo((i/n_colors)) for i in range(n_colors)]
    colors = (np.array(colors)[:,:3] * 255).astype('uint8')

    # gray for index 0
    colors = np.concatenate((np.array([[120, 120, 120]], 'uint8'), colors), 0)
    return colors[labels.ravel()].reshape(labels.shape + (3,))


def bbox_overlap(a, b):
    # (y, x, height, width)

    overlap_y = max(0, (a[0] + a[2] - b[0]) if a[0] < b[0] else (b[0] + b[2] - a[0]))
    overlap_x = max(0, (a[1] + a[3] - b[1]) if a[1] < b[1] else (b[1] + b[3] - a[1]))
    return overlap_y * overlap_x


def imread(filename, grayscale=False):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    img = Image.open(filename)

    if grayscale:
        img = img.convert('L')

    return to_tensor(img)
