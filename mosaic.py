from typing import Any
from abc import ABCMeta
from abc import abstractmethod

import torch
import numpy as np
from more_itertools import grouper
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


class MosaicBatchTransforms(torch.nn.Module, metaclass=ABCMeta): 
    """
    Abstract base class for transforms used in implementing the Mosaic augmentation.
    """

    def __init__(self, min_size: int = 1, disable_warnings: bool = True) -> None:
        super().__init__()
        self.min_size = min_size
        self.disable_warnings = disable_warnings

    @abstractmethod
    def forward(
        self,
        images: tuple[tv_tensors.Image],
        targets: tuple[dict[str, tv_tensors.BoundingBoxes | Any]],
    ) -> tuple[tuple[tv_tensors.Image], tuple[dict[str, torch.Tensor]]]:
        pass

    def validate_forward_inputs(
        self,
        images: tuple[tv_tensors.Image],
        targets: tuple[dict[str, tv_tensors.BoundingBoxes | Any]],
    ) -> None:
        if len(images) != len(targets):
            raise ValueError(f'Unequal number of images and targets: {len(images)} and {len(targets)}.')
        if len(images) < self.min_size:
            raise ValueError(f'Batch size must be at least of the size 4. Given: {len(images)}.')
        if not all(isinstance(image, torch.Tensor) for image in images):
            raise ValueError(f'All images must be Tensors. Found: {set(type(image) for image in images)}.')
        if not all('boxes' in target for target in targets):
            raise ValueError('All targets must have a "boxes" key.')
        if not all(isinstance(target['boxes'], torch.Tensor) for target in targets):
            raise ValueError(f'All bounding boxes must be Tensors. Found: {set(type(target["boxes"]) for target in targets)}.')

        if not self.disable_warnings and len(images) % self.min_size != 0:
            print(f'Warning: batch size should be divisible by {self.min_size} for Mosaic.')


class SquarePad(MosaicBatchTransforms):

    def __init__(self, centre_pad: bool = False, fill_value: int = 114) -> None:
        """
        Pad images to square. If batch size is not divisible by 4, the remainder of images
        will not be transformed.

        Parameters
        ----------
        centre_pad: bool
            If `centre_pad` is `False`, images will be padded "on the outside", i.e., 
            such that no padding will be around the centre of mosaic. Otherwise, padding
            will be equal on all sides of individual images. Defaults to `False`.
        fill_value: int 
            Value to pad with. Defaults to `114` following:
            https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py#L843
        """
        super().__init__()
        self.centre_pad = centre_pad 
        self.fill_value = fill_value

    def forward(
        self,
        images: tuple[tv_tensors.Image],
        targets: tuple[dict[str, tv_tensors.BoundingBoxes | Any]],
    ) -> tuple[tuple[tv_tensors.Image], tuple[dict[str, torch.Tensor]]]:
        self.validate_forward_inputs(images, targets)
        images, targets = list(images), list(targets)
        for batch_idx in grouper(range(len(images)), n=4, incomplete='ignore'):
            for i, pos in zip(batch_idx, ['tl', 'tr', 'bl', 'br']):
                _, h, w = images[i].shape 
                if h == w:
                    continue
                max_wh = np.max([w, h]) 
                p_top, p_left = [(max_wh-s)//2 for s in (h, w)]
                p_bottom, p_right = [max_wh-(s+pad) for s, pad in zip((h, w), (p_top, p_left))]
                # padding values: (left, top, right, bottom)
                if self.centre_pad:
                    padding = (p_left, p_top, p_right, p_bottom)
                    targets[i]['boxes'][:, [0, 2]] += padding[0]
                    targets[i]['boxes'][:, [1, 3]] += padding[1]
                else:
                    if pos == 'tl':
                        padding = (p_left+p_right, p_top+p_bottom, 0, 0)
                        targets[i]['boxes'][:, [0, 2]] += padding[0]
                        targets[i]['boxes'][:, [1, 3]] += padding[1]
                    elif pos == 'tr':
                        padding = (0, p_top+p_bottom, p_left+p_right, 0)
                        targets[i]['boxes'][:, [1, 3]] += padding[1]
                    elif pos == 'bl':
                        padding = (p_left+p_right, 0, 0, p_top+p_bottom)
                        targets[i]['boxes'][:, [0, 2]] += padding[0]
                    elif pos == 'br':
                        padding = (0, 0, p_left+p_right, p_top+p_bottom)
                images[i] = F.pad(images[i], padding, self.fill_value, 'constant')
             
        return tuple(images), tuple(targets)


class Resize(MosaicBatchTransforms):

    def __init__(self, output_size: int) -> None:
        """
        Resize image such that the greater side is equal to the `output_size`.
        """
        super().__init__()
        self.output_size = output_size
    
    def forward(
        self,
        images: tuple[tv_tensors.Image],
        targets: tuple[dict[str, tv_tensors.BoundingBoxes | Any]],
    ) -> tuple[tuple[tv_tensors.Image], tuple[dict[str, torch.Tensor]]]:
        self.validate_forward_inputs(images, targets)
        images, targets = list(images), list(targets)
        for i, image in enumerate(images):
            _, h, w = image.shape 
            max_wh = np.max([w, h]) 
            resize_factor = self.output_size / max_wh
            images[i] = F.resize(images[i], self.output_size, antialias=True)
            targets[i]['boxes'] = targets[i]['boxes'].to(torch.float32)
            targets[i]['boxes'] *= resize_factor
            targets[i]['area'] = targets[i]['area'].to(torch.float32)
            targets[i]['area'] *= resize_factor
             
        return tuple(images), tuple(targets)


class RandomCrop(MosaicBatchTransforms):

    def __init__(self, output_size: int | tuple[int, int] | None, bbox_removal_threshold: float | None = None) -> None:
        """
        Randomly crop the image to size.

        Parameters
        ----------
        output_size: int | tuple[int, int] | None
            The width and height of the cropped image. If an `int` is passed 
            the output will be a square. The `output_size` can be `None`, but then 
            the output size must be passed as the 3rd argument in the `forward` call.
        bbox_removal_threshold: float, optional
            If `bbox_removal_threshold` is a `float`, then bounding boxes with 
            `[area after crop] / [area before crop] < bbox_removal_threshold` will be removed. 
            If `bbox_removal_threshold <= 0.0` or `None`, then no bounding boxes will be 
            removed save for those completely outside of the cropped image.

        Raises
        ----------
        AssertionError
            If in `forward` call both `self.output_size` and passed `output_size` are `None`.
        """
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.bbox_removal_threshold = bbox_removal_threshold

    def forward(
        self,
        images: tuple[tv_tensors.Image],
        targets: tuple[dict[str, tv_tensors.BoundingBoxes | Any]],
        output_size: int | tuple[int, int] | None = None,
    ) -> tuple[tuple[tv_tensors.Image], tuple[dict[str, torch.Tensor]]]:
        assert self.output_size is not None and output_size is not None

        # override output size if provided
        output_size = self.output_size if output_size is None else output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.validate_forward_inputs(images, targets)
        images, targets = list(images), list(targets)
        for i, (image, target) in enumerate(zip(images, targets)):
            # crop the image
            _, h, w = image.shape 
            if w == output_size[0] and h == output_size[1]:
                continue
            x = np.random.randint(0, w-output_size[0]) if w-output_size[0] else 0
            y = np.random.randint(0, h-output_size[1]) if h-output_size[1] else 0
            images[i] = image[:, y:y+output_size[1], x:x+output_size[0]]
            # adjust targets
            target['boxes'] = target['boxes'].to(torch.float32)
            target['boxes'][:, [0, 2]] -= x
            target['boxes'][:, [1, 3]] -= y
            new_target = []
            keys = ['boxes', 'labels', 'iscrowd', 'area']
            for values in zip(*[target[key] for key in keys]):
                bboxes = {key: value for key, value in zip(keys, values)}
                # remove cropped out targets 
                if torch.all(bboxes['boxes'][[0, 2]] < 0) or torch.all(bboxes['boxes'][[1, 3]] < 0):
                    continue
                if torch.all(bboxes['boxes'][[0, 2]] > output_size[0]) or torch.all(bboxes['boxes'][[1, 3]] > output_size[1]):
                    continue
                # adjust the boxes to be contained within the image
                old_area = (bboxes['boxes'][2]-bboxes['boxes'][0])*(bboxes['boxes'][3]-bboxes['boxes'][1])
                bboxes['boxes'] = torch.where(bboxes['boxes'] < 0, 0, bboxes['boxes'])
                bboxes['boxes'][[0, 2]] = torch.where(
                    bboxes['boxes'][[0, 2]] > output_size[0],
                    float(output_size[0]),
                    bboxes['boxes'][[0, 2]]
                )
                bboxes['boxes'][[1, 3]] = torch.where(
                    bboxes['boxes'][[1, 3]] > output_size[1],
                    float(output_size[1]),
                    bboxes['boxes'][[1, 3]]
                )
                bboxes['area'] = (bboxes['boxes'][2]-bboxes['boxes'][0])*(bboxes['boxes'][3]-bboxes['boxes'][1])
                # remove boxes smaller than X% of old area
                if self.bbox_removal_threshold and bboxes['area'] / old_area < self.bbox_removal_threshold:
                    continue
                new_target.append(bboxes)
            if len(new_target) == 1:
                new_target = new_target[0]
                new_target['boxes'] = new_target['boxes'].reshape(1, -1)
            else:
                new_target = {
                    key: torch.cat([val[key].reshape(1) for val in new_target]) if key != 'boxes'
                    else torch.cat([val[key].reshape(1, -1) for val in new_target])
                    for key in keys 
                }
            new_target['image_id'] = target['image_id']
            targets[i] = new_target

        return tuple(images), tuple(targets)


class IntelligentSquareResize(MosaicBatchTransforms):

    def __init__(
        self,
        output_size: int,
        max_aspect_ratio: float | None = None,
        crop_to_square: bool = False,
        centre_pad: bool = False,
        bbox_removal_threshold: float | None = None,
    ) -> None:
        """
        Resize padded and/or cropped images to minimise the amount of padding 
        in the final mosaic.

        Parameters
        ----------
        output_size: int
            The width and height of each individual image in the mosaic 
            (it must be a square, hence only `int` is allowed).
        max_aspect_ratio: float, optional
            If a `float`, then images with aspect ratio (defined as the greater side
            divided by the lesser side, thus, always greater than unity) greater than 
            `max_aspect_ratio` will be cropped to square before any other transform
            is applied.
        crop_to_square: bool
            If `True`, all images will be cropped to square before any other transform 
            is applied. If `True`, `max_aspect_ratio` is ignored. Defaults to `False`.
        centre_pad: bool
            See `SquarePad`.
        bbox_removal_threshold: float, optional
            See `RandomCrop`.
        """
        super().__init__()
        self.output_size = output_size
        self.max_aspect_ratio = max_aspect_ratio
        self.crop_to_square = crop_to_square
        self.centre_pad = False if crop_to_square else centre_pad
        self.bbox_removal_threshold = bbox_removal_threshold

        if not self.crop_to_square and max_aspect_ratio is None:
            raise ValueError(f'Both crop_to_square and max_aspect_ratio are turned off.')

        self.pad_and_resize = v2.Compose([SquarePad(self.centre_pad), Resize(self.output_size)])
        self.random_crop = RandomCrop(-1, self.bbox_removal_threshold)

    def forward(
        self,
        images: tuple[tv_tensors.Image],
        targets: tuple[dict[str, tv_tensors.BoundingBoxes | Any]],
    ) -> tuple[tuple[tv_tensors.Image], tuple[dict[str, torch.Tensor]]]:
        self.validate_forward_inputs(images, targets)
        images, targets = list(images), list(targets)
        for i, (image, target) in enumerate(zip(images, targets)):
            _, h, w = image.shape
            if self.crop_to_square:
                if h == w: 
                    continue
                output_size = (min(h, w), min(h, w))
                image, target = self.random_crop((image,), (target,), output_size)
                images[i], targets[i] = image[0], target[0]
                continue
            aspect_ratio = max(h, w) / min(h, w)
            if self.max_aspect_ratio and aspect_ratio > self.max_aspect_ratio:
                greater_size = int(self.max_aspect_ratio * min(h, w))
                output_size = (greater_size, h) if w > h else (w, greater_size)
                image, target = self.random_crop((image,), (target,), output_size)
                images[i], targets[i] = image[0], target[0]
        images, targets = self.pad_and_resize(images, targets)

        return tuple(images), tuple(targets)


class TypeConverter(MosaicBatchTransforms):
    """
    Converts images to `torch.float32`, as well as, target bounding boxes 
    to `tv_tensors.BoundingBoxes` and target area to `torch.int32`.
    """

    def forward(
        self,
        images: tuple[tv_tensors.Image],
        targets: tuple[dict[str, tv_tensors.BoundingBoxes | Any]],
    ) -> tuple[tuple[tv_tensors.Image], tuple[dict[str, torch.Tensor]]]:
        self.validate_forward_inputs(images, targets)
        images, targets = list(images), list(targets)
        for i in range(len(images)):
            images[i] = images[i].to(torch.float32)
            targets[i]['boxes'] = tv_tensors.BoundingBoxes(
                targets[i]['boxes'],
                format='XYXY',
                canvas_size=tuple(images[i].shape[1:]),
            )
            targets[i]['area'] = targets[i]['area'].to(torch.int32)

        return tuple(images), tuple(targets)


class Mosaic(MosaicBatchTransforms):

    def __init__(
        self,
        output_size: int = 1_000,
        min_possible_image_area: float = 0.25,
        centre_pad: bool = False,
        crop: bool = True,
        bbox_removal_threshold: bool | float = 0.05,
        intelligent_resize: bool = True,
        max_aspect_ratio: float | None = None,
        crop_to_square: bool = False,
    ) -> None:
        """
        Mosaic data augmentation.

        Parameters
        ----------
        output_size: int
            The size of the final image. Defaults to `1000`.
        min_possible_image_area: float
            The minimal fraction of any of the four images to be visiable in the cropped mosaic.
            This is added in order to avoid excessive cropping of any of the images.
            It is used to calculate pre-cropping size of the grid. Deafults to `0.25`.

            For example, if `output_size = 1000` and `min_possible_image_area = 0.25`, then, first,
            a `1500x1500` grid will be constructed (each image will be resized to `750x750`) and 
            then it will be randomly cropped to `1000x1000`.
            
            If `crop = False`, then a `1000x1000` grid will be constructed straight away.
        centre_pad: bool
            If `True`, the four images will be padded with equal padding on each side as needed 
            to create a square. Otherwise, the padding will be added such that no image has 
            padding separating it from the centre of the mosaic. Defaults to `False`.
            See `SquarePad`.
        crop: bool 
            If `True`, a larger grid will be constructed first before randomly cropping
            to the required size. Defaults to `True`.
        bbox_removal_threshold: bool | float
            Parameter passed to `RandomCrop`. If not `False` and greater than `0.0`, the bounding
            boxes that after cropping have less than `bbox_removal_threshold` fraction of area 
            visiable in the cropped image will be removed. This is done to avoid edge cases where 
            almost all of the bounding box resides outside of the cropped image. Defaults to `0.05`.
            See `RandomCrop`.
        intelligent_resize: bool
            If `False`, all images will be padded to square and resized. This may result in large amounts 
            of padding visiable in the final mosaic if the dataset has images of various aspect ratios.
            If `True`, images with aspect ratio (defined as the greater side divided by the lesser side,
            thus, always greater than unity) greater than `max_aspect_ratio` will be cropped to square 
            before any other transforms are applied. Defaults to `True`.
        max_aspect_ratio: float, optional
            If a `float`, then images with aspect ratio (defined as the greater side divided by the lesser
            side, thus, always greater than unity) greater than `max_aspect_ratio` will be cropped to
            square before any other transform is applied.
        crop_to_square: bool
            If `True`, all images will be cropped to square before any other transform is applied. 
            If `True`, `max_aspect_ratio` is ignored. Defaults to `False`.
        """
        super().__init__(min_size=4)
        self.output_size = output_size
        self.min_possible_image_area = min_possible_image_area
        self.centre_pad = centre_pad 
        self.crop = crop
        self.bbox_removal_threshold = bbox_removal_threshold
        self.intelligent_resize = intelligent_resize
        self.max_aspect_ratio = max_aspect_ratio  
        self.crop_to_square = crop_to_square 
        # the total size of the grid of 4 images
        self.grid_size = self.output_size * (2 - np.sqrt(self.min_possible_image_area)) if self.crop else self.output_size
        # the target size of an image
        self.images_size = int(self.grid_size / 2)
        # preprocessing transforms
        if self.intelligent_resize:
            self.preprocess = v2.Compose([IntelligentSquareResize(
                output_size=self.images_size,
                max_aspect_ratio=self.max_aspect_ratio,
                crop_to_square=self.crop_to_square,
                bbox_removal_threshold=self.bbox_removal_threshold,
            )])
        else:
            self.preprocess = v2.Compose([SquarePad(self.centre_pad), Resize(self.images_size)])
        # postprocessing transforms
        if crop:
            self.postprocess = v2.Compose([RandomCrop(self.output_size, self.bbox_removal_threshold), TypeConverter()])
        else:
            self.postprocess = v2.Compose([TypeConverter()])
    
    def forward(
        self,
        images: tuple[tv_tensors.Image],
        targets: tuple[dict[str, tv_tensors.BoundingBoxes | Any]],
    ) -> tuple[tuple[tv_tensors.Image], tuple[dict[str, torch.Tensor]]]:
        self.validate_forward_inputs(images, targets)
        # resize the images to fit their quarters
        images, targets = self.preprocess(images, targets)
        # join the images into a grid
        new_images, new_targets = [], []
        for batch_idx in grouper(range(len(images)), n=4, incomplete='ignore'):
            # join images
            img_batch = [images[i] for i in batch_idx]
            top = torch.cat(img_batch[:2], dim=2)
            bottom = torch.cat(img_batch[2:], dim=2)
            grid_images = torch.cat([top, bottom], dim=1)
            new_images.append(grid_images)
            # correct targets to fit the grid
            tar_batch = [targets[i] for i in batch_idx]
            for i, pos in enumerate(['tl', 'tr', 'bl', 'br']):
                if pos == 'tl':
                    continue
                elif pos == 'tr':
                    tar_batch[i]['boxes'][:, [0, 2]] += self.images_size
                elif pos == 'bl':
                    tar_batch[i]['boxes'][:, [1, 3]] += self.images_size
                elif pos == 'br':
                    tar_batch[i]['boxes'][:, [0, 2]] += self.images_size
                    tar_batch[i]['boxes'][:, [1, 3]] += self.images_size
            grid_targets = {}
            for key in tar_batch[0].keys():
                grid_targets[key] = torch.cat([
                    target[key].reshape(1) if len(target[key].shape) == 0 else target[key]
                    for target in tar_batch
                ])
            new_targets.append(grid_targets)
        # crop the images to output size and convert to proper dtypes
        new_images, new_targets = self.postprocess(new_images, new_targets)
        new_images, new_targets = list(new_images), list(new_targets)

        if (rem := divmod(len(images), 4)[1]) != 0:
            new_images += list(images)[-rem:]
            new_targets += list(targets)[-rem:]

        return tuple(new_images), tuple(new_targets)

