import random
import torch
import numpy as np
import numpy as np
from functools import partial

from tralo.log import log
from tralo.utils import resize, images_to_grid, color_label_image


class VisualizerBase(object):

    def __init__(self, data_item, target_size):
        self.data_transformed = None
        self.data_item = data_item
        self.target_size = target_size

    def generate(self, target_size):
        pass

    def plot(self, ax):
        raise NotImplementedError

    def get_info(self, frame, column):
        """ returns text info as tkinter Label """
        from tkinter import Label, W
        label = Label(frame, text='no info available')
        label.grid(column=column + 2, row=3, sticky=(W,))
        return label


class Image(VisualizerBase):
    def __init__(self, data_item, target_size, color=True, channel_dim=0, bgr=False, colormap=None):
        # Frame.__init__(self, parent.mainframe)
        super().__init__(data_item, target_size)

        assert not (color and (colormap is not None)), 'A color image can not be color-mapped'

        self.color = color
        self.channel_dim = channel_dim
        self.bgr = bgr

        assert type(data_item) == np.ndarray

        self.data_item = data_item
        self.generate(target_size)

    def generate(self, target_size=None):

        data_item_view = self.data_item.copy()

        assert data_item_view.ndim in {3, 2}, 'Image must have 3 (color) or 2 (grayscale) dimensions.'

        if self.color:
            if self.channel_dim == 0:
                data_item_view = data_item_view.transpose([1, 2, 0])
            elif self.channel_dim == 1:
                data_item_view = data_item_view.transpose([0, 2, 1])
            elif self.channel_dim == 2:
                data_item_view = data_item_view.transpose([0, 1, 2])

        # correct for negative values
        if -1 <= data_item_view.min() < 0 <= data_item_view.max() <= 1:
            data_item_view += 1
            data_item_view /= 2
        elif -100 <= data_item_view.min() < 0 <= data_item_view.max() <= 100:
            data_item_view -= data_item_view.min()
            data_item_view /= data_item_view.max()
        elif -255 <= data_item_view.min() < 0 <= data_item_view.max() <= 255:
            data_item_view += 255
            data_item_view /= 2

        if self.bgr:
            data_item_view = data_item_view[:, :, [2, 1, 0]]

        if target_size is not None:
            data_item_view = resize(data_item_view, target_size, max_bound=True, channel_dim=2 if len(data_item_view.shape)==3 else 0)
        self.data_transformed = data_item_view
        # return data_item_view

    def as_image(self):
        data_item_view = self.data_transformed.copy()

        if data_item_view.max() <= 1 and data_item_view.dtype.name.startswith('float'):
            data_item_view = np.uint8(data_item_view * 255)
        else:
            data_item_view = np.uint8(data_item_view)

        return data_item_view

    def plot(self, ax):

        import matplotlib.pyplot as plt

        if np.issubdtype(self.data_transformed.dtype, np.floating):
            if self.data_transformed.max() > 1.001:
                self.data_transformed = self.data_transformed / 255

        ax.imshow(self.data_transformed, cmap=None if self.color else plt.get_cmap('gray'))
        ax.axis('off')


class Slices(VisualizerBase):
    """
    `maps_dim` indicates the dimension along which the maps are stored.
    `maps_dim` and `channel_dim` ignore the batch dimension
    """
    def __init__(self, data_item, target_size, maps_dim, slice_labels=None, # maps_dim=None,
                 channel_dim=None, color=False, normalize=False):
        super().__init__(data_item, target_size)

        self.normalize = normalize
        assert maps_dim is not None, 'maps_dim can not be None'
        # assert channel_dim is not None and color or channel_dim is None

        self.cursor = 0
        self.maps_dim = maps_dim
        self.slice_labels = slice_labels
        if slice_labels is not None:
            log.detail('Using slice', len(slice_labels), 'labels:', slice_labels)
        self.frame = None
        self.channel_dim = channel_dim

        self.target_size = target_size
        self.generate(self.target_size)
        # self.draw_slice()

    def plot(self, ax):

        img = self.as_image()
        norm = 255 if 1 < img.max() <= 255 else 1
        img = img / norm

        if img.ndim == 3:
            img = img.transpose([1, 2, 0])
 
        import matplotlib.pyplot as plt
        ax.axis('off')
        ax.imshow(img, cmap=plt.cm.gray)

    def generate(self, target_size):
        """ Resizes to `target size` but also sets channels last and maps first. """
        # treat the images as channels

        transformed_data = []
        for i in range(self.data_item.shape[self.maps_dim]):
            slice_map = np.take(self.data_item, i, self.maps_dim)

            # ipdb.set_trace()
            if self.channel_dim is not None:
                corrected_channel_dim = self.channel_dim - (1 if self.channel_dim > self.maps_dim else 0)
            else:
                corrected_channel_dim = None

            # resized_map = tensor_resize(slice_map, target_size, interpret_as_max_bound=True,
            #                             channel_dim=corrected_channel_dim, keep_channels_last=True)
            resized_map = resize(slice_map, target_size, channel_dim=corrected_channel_dim)

            if corrected_channel_dim == 0:
                resized_map = resized_map.transpose([1,2,0])
            elif corrected_channel_dim is None:
                resized_map = resized_map[:,:,None]

            transformed_data += [resized_map]

        self.transformed_data = np.array(transformed_data)

        if self.normalize:
            self.transformed_data -= self.transformed_data.min()
            self.transformed_data /= self.transformed_data.max()

    def as_image(self):

        grid_image = images_to_grid(self.transformed_data, self.target_size).numpy()

        if len(grid_image.shape) == 3:
            grid_image = grid_image.transpose([2, 0, 1])

        if grid_image.max() <= 1:
            grid_image *= 255

        return grid_image

    def get_visdom_data(self):
        return self.as_image(), 'image'


class DistributionData(VisualizerBase):

    def generate(self, target_size):
        pass

    def __init__(self, data_item, target_size):
        super().__init__(data_item, target_size)

        text = "\ndtype: " + str(data_item.dtype)
        text += "\nsize: " + str(len(data_item))
        text += "\nsum: " + str(data_item.sum())
        text += "\nargmax: " + str(data_item.argmax())
        text += "\nnan/inf: " + str(np.any(np.isnan(data_item)) or np.any(np.isinf(data_item)))
        self.text = text
        assert type(data_item) == np.ndarray

    def as_bar(self):
        return self.data_item

    def plot(self, ax):
        if len(self.data_item) > 200:
            ax.text(0, 0, 'Too large to visualize', fontsize=15)
            ax.axis('off')
            pass
        else:
            ax.bar(np.arange(0, len(self.data_item)), self.data_item, 0.9)


class LabelImage(VisualizerBase):

    def __init__(self, data_item, target_size, maps_dim=None, n_colors=None):
        super().__init__(data_item, target_size)
        self.maps_dim = maps_dim
        self.n_colors = n_colors
        # self.one_hot = one_hot

        self.data_item = data_item
        self.transformed_data = None
        self.target_size = target_size
        self.data_item_view = None

        self.generate(target_size)

    def generate(self, target_size):
        data_item = self.data_item.copy()

        assert type(self.data_item) == np.ndarray
        assert np.issubdtype(self.data_item.dtype, np.integer)

        if data_item.ndim == 3:
            data_item = np.argmax(data_item, axis=0)

        if self.maps_dim is None:
            assert len(data_item.shape) == 2
        else:
            assert len(data_item.shape) == 3 and 0 <= self.maps_dim <= 2

        data_item = data_item.astype('int16')
        self.data_item_view = resize(data_item, target_size,max_bound=True,channel_dim=self.maps_dim, interpolation='nearest')

        if self.maps_dim is not None:
            labels = self.data_item_view.argmax(2)
        else:
            labels = self.data_item_view

        self.data_transformed = color_label_image(labels, self.n_colors)

    def plot(self, ax):

        vmax = 1 if self.data_transformed.max() <= 1 else 255

        ax.imshow(self.data_transformed, vmin=0, vmax=vmax)
        ax.axis('off')



class TextData(VisualizerBase):

    def __init__(self, data_item, target_size):
        super().__init__(data_item, target_size)

    def plot(self, ax):

        import matplotlib.pyplot as plt

        ax.imshow(np.ones((300, 300)) * 0.8, vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
        ax.text(10, 290, str(self.data_item).replace('\n', ' ')[:30] + ('...' if len(str(self.data_item)) > 30 else ''), fontsize=8)
        ax.axis('off')

    def as_text(self):
        return str(self.data_item)


# Shortcuts
ImageGrayscale = partial(Image, color=False)
DenseM = partial(Slices, maps_dim=0, color=False)
Video = partial(Slices, channel_dim=0, maps_dim=1, color=True)
Video_maps_first = partial(Slices, channel_dim=1, maps_dim=0, color=True)
VideoGrayscale = partial(Slices, channel_dim=0, maps_dim=1, color=False)


def guess_element(element, target_size):
    """ some heuristics for guessing the data type based on a sample only """

    if isinstance(element, (np.int_, int, float)):
        return TextData(str(element), target_size)

    original_element = element

    if isinstance(element, torch.Tensor):
        element = element.cpu().detach()

    element = np.array(element)
    shape = element.shape if hasattr(element, 'shape') else ()
    ndim = element.ndim if hasattr(element, 'shape') else 0

    # with shape
    if ndim == 4:

        if shape[0] == 1 and shape[3] == 3:
            return Image(element[0], target_size, channel_dim=2)

        elif shape[1] in {1, 3}:
            return Slices(element, target_size, channel_dim=1, maps_dim=0)

        elif shape[0] in {1, 3}:
            return Slices(element, target_size, channel_dim=0, maps_dim=1)

    elif ndim == 3:

        if shape[0] == 1:
            return ImageGrayscale(element[0], target_size)

        elif shape[0] == 3:  # rgb image
            return Image(element, target_size, channel_dim=0)

        elif shape[2] == 3:  # rgb image
            return Image(element, target_size, channel_dim=2)

        elif shape[0] > 3 and shape[1] == shape[2]:
            return Slices(element, target_size, maps_dim=0)            

        elif shape[0] > 3 and shape[0] == shape[1]:  # sequence
            return Slices(element, target_size, maps_dim=2)    

        elif shape[0] > 3:
            return Slices(element, target_size, maps_dim=0)
    
    elif ndim == 2:

        if np.issubdtype(element.dtype, np.integer):
            return LabelImage(element, target_size, maps_dim=None)

        elif np.issubdtype(element.dtype, np.floating):
            return Image(element, target_size, color=False)

        elif shape[0] > 10:
            return ImageGrayscale(element, target_size)

    elif ndim == 1:

        if shape[0] == 0:
            return TextData('Empty', target_size)

        if np.issubdtype(type(element[0]), np.integer):
            return TextData(str(element[0]), target_size)

        # return DistributionData(element, target_size)

    elif ndim == 0:
        if type(original_element) == str:
            return TextData(original_element, target_size)

    raise ValueError('Failed to guess visualizer')


def guess_visualizer(samples, target_size, types=None):

    vis_list, info = [], []
    types = types if types is not None else [None for _ in samples]

    for s, type_name in zip(samples, types):

        if type(type_name) == str:
            vis_list += [locals()[type_name](s, target_size)]
        elif callable(type_name):
            vis_list += [type_name(s, target_size)]
        else:
            try:
                vis_list += [guess_element(s, target_size)]
            except ValueError as e:
                vis_list += [TextData(str(e), target_size)]

    return vis_list, None


def _plot_visualizer(vi, elements, height, titles=None):

    import matplotlib.pyplot as plt

    eles = list(enumerate(vi)) if elements is None else [(j, vi[j]) for j in elements]

    w, h = min(16, len(eles) * height), min(height, 16 / len(eles))
    fig, ax = plt.subplots(1, len(eles), figsize=(w, h))

    for k, (j, v) in enumerate(eles):
        axis = ax[k] if len(eles) > 1 else ax
        v.plot(axis)
        # ax[i].text(0, 0, , fontsize=15)
        if titles is not None:
            axis.set_title(titles[j], fontsize=9)


def _prepare_image(img):
    if type(img) == torch.Tensor:
        img = img.detach().cpu().numpy()

    if img.shape[0] == 1:
        img = img[0]
    elif img.shape[0] == 3:
        img = img.transpose([1, 2, 0])

    return img


def plot_image(image, axis=None, figsize=(5, 5)):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(ncols=1, figsize=figsize) if axis is None else (None, axis)
    ax.axis('off')
    ax.imshow(_prepare_image(image))


def plot_image_stack(images, figsize=(15, 5), cmap='gray', axis=None, no_stats=False):

    import matplotlib.pyplot as plt

    vmin, vmax = min([img.min() for img in images]), max([img.max() for img in images])
    fig, ax = plt.subplots(ncols=len(images), figsize=figsize) if axis is None else (None, axis)

    log.detail(f'min {min([i.min() for i in images])}, max {max([i.max() for i in images])}')

    for i in range(len(images)):

        img = images[i]
        img = _prepare_image(img)

        this_ax = ax[i] if len(images) > 1 else ax
        this_ax.imshow(img, cmap=plt.cm.get_cmap(cmap), vmin=vmin, vmax=vmax)
        this_ax.axis('off')
    
    if fig is not None:
        fig.tight_layout()

    return fig


def plot_data(dataset, end=1, start=0, shuffle=False, height=2.5, elements=None, types=None):


    if type(dataset) in {torch.Tensor, np.ndarray}:
        vi, _ = guess_visualizer([dataset], (300, 300), types=types)
        _plot_visualizer(vi, elements, height)
    else:

        if not shuffle:
            indices = range(start, end)
        else:
            indices = list(range(len(dataset)))
            random.shuffle(indices)

            indices = indices[start: end]

        for i in indices:
            sample = dataset[i]
            if len(sample) == 2 and type(sample[0]) in {list, tuple} and type(sample[1]) in {list, tuple}:
                items = [s for s in sample[0]] + [s for s in sample[1]]
                titles = ['in' for _ in sample[0]] + ['out' for _ in sample[1]]
            else:
                items = sample
                titles = [''] * len(sample)
            
            vi, _ = guess_visualizer(items, (300, 300), types=types)
            
            mins, maxs, shapes = ['']*len(items), ['']*len(items), ['']*len(items)
            for j, x in enumerate(items):
            
                try:
                    shapes[j] = tuple(x.shape)
                    mins[j] = round(float(x.min()), 3)
                    maxs[j] = round(float(x.max()), 3)
                except (RuntimeError, AttributeError):
                    pass
                
            dtypes = [x.dtype if hasattr(x, 'dtype') else '' for x in items]
            titles = [f'{t} {s}\n{dt} {mn}-{mx}' for t, s, dt, mn, mx in zip(titles, shapes, dtypes, mins, maxs)]

            _plot_visualizer(vi, elements, height, titles)