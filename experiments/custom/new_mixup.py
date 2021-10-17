import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

# from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
# from ..builder import PIPELINES
from mmdet.datasets import PIPELINES

import random

'''
1. target이미지안에서 bounding box의 개수가 [1, 3]가 되게 해줌
2. (target이미지 * 0.3 + original이미지 * 0.7) 로 변경 -> 취소
3. prob 인자값 줌
'''
@PIPELINES.register_module()
class NewMixUp:
    """MixUp data augmentation.

    .. code:: text
                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The mixup transform steps are as follows::

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
           Default: (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
           Default: (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
           Default: 0.5.
        pad_val (int): Pad value. Default: 114.
        max_iters (int): The maximum number of iterations. If the number of
           iterations is greater than `max_iters`, but gt_bbox is still
           empty, then the iteration is terminated. Default: 15.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 5.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Default: 20.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 ratio_range=(0.5, 1.5),
                 flip_ratio=0.5,
                 pad_val=114,
                 max_iters=15,
                 min_bbox_size=5,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 prob=0.5): # prob추가
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.prob = prob

    def __call__(self, results):
        """Call function to make a mixup of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mixup transformed.
        """

        results = self._mixup_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(dataset))
            '''
            gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
            # if len(gt_bboxes_i) != 0:
            if 0 < len(gt_bboxes_i) and len(gt_bboxes_i) <= 3:
                break
            '''
            try:
                gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
                # if len(gt_bboxes_i) != 0:
                if 0 < len(gt_bboxes_i) and len(gt_bboxes_i) <= 3:
                    break
            except IndexError:
                print('Error!!!!!!!!!!!!! index:', index)
        # print(f'index: {index}')
        return index


    def _mixup_transform(self, results):
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        # print('result: ', results)
        '''
        result
        {
            'img_info':{},
            'ann_info':{},
            ...,
            'mix_results':[{
                'img_info':{},
                'ann_info':{},
                ...
            }]
        }
        '''
        if random.random() < self.prob:
            return results

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # 0.5의 확률로 적용
            # empty bbox
            return results

        if 'scale' in results:
            self.dynamic_scale = results['scale']

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w,
                                          target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes[:, 0::2] = np.clip(
            retrieve_gt_bboxes[:, 0::2] * scale_ratio, 0, origin_w)
        retrieve_gt_bboxes[:, 1::2] = np.clip(
            retrieve_gt_bboxes[:, 1::2] * scale_ratio, 0, origin_h)

        if is_filp:
            retrieve_gt_bboxes[:, 0::2] = (
                origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1])

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
            cp_retrieve_gt_bboxes[:, 0::2] - x_offset, 0, target_w)
        cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
            cp_retrieve_gt_bboxes[:, 1::2] - y_offset, 0, target_h)
        keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T,
                                                cp_retrieve_gt_bboxes.T)

        # 8. mix up
        if keep_list.sum() >= 1.0:
            ori_img = ori_img.astype(np.float32)
            mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(
                np.float32)
            # mixup_img = 0.7 * ori_img + 0.3 * padded_cropped_img.astype(
            #     np.float32)

            retrieve_gt_labels = retrieve_results['gt_labels'][keep_list]
            retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]
            mixup_gt_bboxes = np.concatenate(
                (results['gt_bboxes'], retrieve_gt_bboxes), axis=0)
            mixup_gt_labels = np.concatenate(
                (results['gt_labels'], retrieve_gt_labels), axis=0)

            results['img'] = mixup_img
            results['img_shape'] = mixup_img.shape
            results['gt_bboxes'] = mixup_gt_bboxes
            results['gt_labels'] = mixup_gt_labels

        return results

    def _filter_box_candidates(self, bbox1, bbox2):
        """Compute candidate boxes which include following 5 things:

        bbox1 before augment, bbox2 after augment, min_bbox_size (pixels),
        min_area_ratio, max_aspect_ratio.
        """

        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size)
                & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range})'
        repr_str += f'flip_ratio={self.flip_ratio})'
        repr_str += f'pad_val={self.pad_val})'
        repr_str += f'max_iters={self.max_iters})'
        repr_str += f'min_bbox_size={self.min_bbox_size})'
        repr_str += f'min_area_ratio={self.min_area_ratio})'
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio})'
        return repr_str
