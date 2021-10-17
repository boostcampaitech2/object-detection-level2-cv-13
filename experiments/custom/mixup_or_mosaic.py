# import copy
# import inspect
# import math
# import warnings

# import cv2
# import mmcv
# import numpy as np
# from numpy import random

# # from mmdet.core import PolygonMasks
# from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
# # from ..builder import PIPELINES
# from mmdet.datasets import PIPELINES

# import random


# @PIPELINES.register_module()
# class MixUpOrMosaic:
#     def __init__(self,
#                  img_scale=(640, 640),
#                  center_ratio_range=(0.5, 1.5),
#                  ratio_range=(0.5, 1.5),
#                  flip_ratio=0.5,
#                  pad_val=114,
#                  max_iters=15,
#                  min_bbox_size=5,
#                  min_area_ratio=0.2,
#                  max_aspect_ratio=20,
#                  prob=0.8): # 0.5*prob : Mosaic / 0.5*(1-prob) : MixUp / 50% : None
#         assert isinstance(img_scale, tuple)
#         assert 0 <= prob <= 1

#         self.dynamic_scale = img_scale
#         self.center_ratio_range = center_ratio_range
#         self.ratio_range = ratio_range
#         self.flip_ratio = flip_ratio
#         self.pad_val = pad_val
#         self.max_iters = max_iters
#         self.min_bbox_size = min_bbox_size
#         self.min_area_ratio = min_area_ratio
#         self.max_aspect_ratio = max_aspect_ratio
#         self.prob = prob

#     def __call__(self, results):
#         random_val = random.random()
#         if random_val > 0.5:
#             pass    
#         elif random_val < 0.5 * (1 - self.prob):
#             results = self._mixup_transform(resultss)
#         else:
#             results = self._mosaic_transform(results)
#         return results

    
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
1. prob 인자값 줌
2. MixUp/Mosaic/None 모드는 get_indexes에서 결정
3. 모드 확인은 
   results['mix_results'] == 0 : None
   results['mix_results'] == 1 : MixUp
   results['mix_results'] == 3 : Mosaic
'''
@PIPELINES.register_module()
class MixUpOrMosaic:
    def __init__(self,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 ratio_range=(0.5, 1.5),
                 flip_ratio=0.5,
                 pad_val=114,
                 max_iters=15,
                 min_bbox_size=5,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 prob=0.8): # 0.5*prob: Mosaic / 0.5*(1-prob): MixUp / 0.5: None
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1
        self.dynamic_scale = img_scale
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.prob = prob

    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """

        results = self._mixup_or_mosaic_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list or int or None: index.
        """
        random_val = random.random()
        # print('random_val:', random_val)

        index = None
        # 0.5*prob: Mosaic / 0.5*(1-prob): MixUp / 0.5: None
        if random_val > 0.5 :
            index = []
        elif random_val < 0.5*(1-self.prob):
            for i in range(self.max_iters):
                index = random.randint(0, len(dataset)-1)
                '''
                gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
                # if len(gt_bboxes_i) != 0:
                if 0 < len(gt_bboxes_i) and len(gt_bboxes_i) <= 3:
                    break
                '''
                gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
                # if len(gt_bboxes_i) != 0:
                if 0 < len(gt_bboxes_i) and len(gt_bboxes_i) <= 3:
                    break
        else:
            index = [random.randint(0, len(dataset) - 1) for _ in range(3)]
        # print('index:', index)
        return index

    def _mixup_or_mosaic_transform(self, results):
        '''
        results['mix_results'] == 0 : None
        results['mix_results'] == 1 : MixUp
        results['mix_results'] == 3 : Mosaic
        '''
        assert 'mix_results' in results
        mode_int = len(results['mix_results'])
        # print('mode_int: ', mode_int)
        assert mode_int in [0, 1, 3]

        if mode_int == 0:
            return results
        elif mode_int == 1:
            return self._mixup_transform(results)
        elif mode_int == 3:
            return self._mosaic_transform(results)

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        # if random.random() < self.prob:
        #     return results
        # print(results['mix_results'][0])

        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                             2 * self.img_scale[1])
            mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                             2 * self.img_scale[0])
            mosaic_labels = np.concatenate(mosaic_labels, 0)

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['ori_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """

        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

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
        # if random.random() < self.prob:
        #     return results

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
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range})'
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

