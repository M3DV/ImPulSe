from copy import deepcopy

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform


INTERPOLATIONS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
}

def _fist_index_not_zero(volume):
    slice_sum = volume.sum(-1).sum(-1)
    lenth = len(slice_sum)
    for i in range(lenth):
        if slice_sum[i]>0:
            break
    first = i
    for i in range(lenth-1, -1, -1):
        if slice_sum[i]>0:
            break        
    last = i
    return [first, last]


def get_bbox_from_mask(volume):
    crop = []
    crop.append(_fist_index_not_zero(volume))
    crop.append(_fist_index_not_zero(volume.transpose(1,0,2)))
    crop.append(_fist_index_not_zero(volume.transpose(2,0,1)))
    return np.array(crop)


def _resample(image, target_spacing, target_shape, interpolation):
    # set up resampling parameters
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(INTERPOLATIONS[interpolation])
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_shape)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    # execute the resampling
    image = resampler.Execute(image)

    return image


def _resample_to_shape(arr, target_shape, interpolation):
    # convert np.ndarray to sitk.Image
    original_type = arr.dtype
    image = sitk.GetImageFromArray(arr.astype(float))

    # calculate the target spacing, assuming the original spacing is 1x1x1
    target_spacing = tuple([arr.shape[i] / target_shape[i]
        for i in range(len(target_shape))])

    # reverse spacing and shape to xyz format
    target_spacing = tuple(reversed(target_spacing))
    target_shape = tuple(reversed(target_shape))

    # resampling
    image = _resample(image, target_spacing, target_shape, interpolation)

    # convert sitk.Image back to np.ndarray
    new_arr = sitk.GetArrayFromImage(image).astype(original_type)

    return new_arr


class GetLungBbox:
        
    def __call__(self, data):
 
        data["lung_bbox"] = get_bbox_from_mask(data["lungsegment"])
        return data


class CropLung:

    def __call__(self, data):
        bbox = data["lung_bbox"]
        data["raw_res"] = data["lungsegment"].shape
        data["raw_lungsegment"] = deepcopy(data["lungsegment"])
        data["raw_airway"] = deepcopy(data["airway"])
        data["raw_artery"] = deepcopy(data["artery"])
        data["raw_vein"] = deepcopy(data["vein"])
        keys = ["airway", "artery", "vein", "lungsegment"]
        for k in keys:
            data[k] = data[k][
                bbox[0, 0]:bbox[0, 1] + 1,
                bbox[1, 0]:bbox[1, 1] + 1,
                bbox[2, 0]:bbox[2, 1] + 1
            ]

        return data


class GetLobe:

    def __init__(self):
        self.lobe_mapping = {
        (1, 3): 1,
        (4, 5): 2,
        (6, 10): 3,
        (11, 14): 4,
        (15, 18): 5
        }
    
    def __call__(self, data):
        if "lobe" not in data:
            data["lobe"] = np.zeros_like(data["lungsegment"])
            for seg_rng, lobe_idx in self.lobe_mapping.items():
                data["lobe"][
                    np.logical_and(data["lungsegment"] >= seg_rng[0],
                    data["lungsegment"] <= seg_rng[1])
                ] = lobe_idx

        return data


class Resample:

    def __init__(self, configs):
        self.configs = configs

    def __call__(self, data):
        data["original_res"] = data["lungsegment"].shape
        for k in self.configs.keys():
            if k in ["airway", "artery"]:
                data[f"original_{k}"] = deepcopy(data[k])
            data[k] = _resample_to_shape(data[k], self.configs[k]["size"],
                self.configs[k]["interp"])

        return data


class ClipValue:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data):
        data["image"] = np.clip(data["image"], self.min_val, self.max_val)

        return data


class MinMaxNormalize:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data):
        data["image"] = (data["image"] - self.min_val)\
            / (self.max_val - self.min_val)

        return data


class OnehotEncode:

    def __init__(self, key, num_classes):
        self.key = key
        self.num_classes = num_classes
    
    def __call__(self, data):
        d, h, w = data[self.key].shape
        flattened = data[self.key].reshape(-1)
        onehot_flattened = np.eye(self.num_classes)[flattened]
        data[self.key] = onehot_flattened.reshape((self.num_classes, d, h, w))

        return data


class ConcatInputs:

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        data["inputs"] = np.concatenate([data[k] if data[k].ndim == 4
            else data[k][np.newaxis] for k in self.keys])

        return data


class SampleGrid:

    def __init__(self, resolution=None, mode="uniform"):
        self.resolution = resolution
        if resolution is None:
            self.reg_grid = None
        else:
            axes = [torch.linspace(-1, 1, resolution[i]) for i in range(3)]
            self.reg_grid = torch.stack(torch.meshgrid(axes[2], axes[1],
                axes[0], indexing="ij"), -1)

        assert mode in ["uniform", "perturbed", "random", "bav"]
        self.mode = mode
        if self.mode == "perturbed":
            self.sampler = Uniform(-(2 / resolution), 2 / resolution)
        elif self.mode == "random" or self.mode == "bav":
            self.sampler = Uniform(-1, 1)

    def __call__(self, data):
        if self.resolution is None:
            resolution = data["lungsegment"].shape
            axes = [torch.linspace(-1, 1, resolution[i]) for i in range(3)]
            self.reg_grid = torch.stack(torch.meshgrid(axes[2], axes[1],
                axes[0], indexing="ij"), -1)
        if self.mode == "uniform":
            data["grids"] = self.reg_grid
        elif self.mode == "perturbed":
            pertubations = self.sampler.sample(self.reg_grid.size())
            data["grids"] = self.reg_grid + pertubations
        elif self.mode == "random":

            data["grids"] = self.sampler.sample(self.reg_grid.size())
        elif self.mode == "bav":

            bav_mask = np.logical_or(data["airway"] > 0,
            np.logical_or(data["artery"] > 0, data["vein"] > 0))
            vol_shape = np.array(data["airway"].shape)

            bav_coords = np.argwhere(bav_mask)

            # calculate total number of index for the grid
            n_pts = self.reg_grid.nelement()
            # calculate the number of points
            n_pts = n_pts // 3
            # properation of points sampled in BAV space
            n_bav_pts = n_pts * 0.2
            n_bav_pts = int(n_bav_pts)
            n_random_pts = n_pts - n_bav_pts

            # random sampled points
            bav_samples = bav_coords[np.random.randint(0, bav_coords.shape[0], (n_bav_pts, ))]
            bav_samples = bav_samples / (vol_shape - 1) * 2 - 1
            bav_samples = torch.from_numpy(bav_samples).float()
            bav_samples = torch.flip(bav_samples, dims=[-1])

            random_samples = self.sampler.sample(torch.tensor([n_random_pts, 3]))

            samples = torch.cat((bav_samples, random_samples), dim=0)
            samples = samples.reshape(self.reg_grid.size())

            data["grids"] = samples
        else:
            raise ValueError(f"Unrecognized sampling mode {self.mode}.")

        return data


class SampleTarget:

    def __init__(self, key, interpolation="nearest"):
        self.key = key
        self.interpolation = interpolation

    def __call__(self, data):
        data[self.key] = torch.from_numpy(data[self.key]).float()
        if data["grids"].size()[-2:0:-1] == data[self.key].size():
            data["targets"] = data[self.key][None, None]
        else:
            data["targets"] = F.grid_sample(data[self.key][None, None],
                data["grids"][None], self.interpolation, align_corners=True)

            data["targets"] = data["targets"].permute(0, 1, 4, 3, 2)


        return data


class ToTensor:

    def __call__(self, data):
        data["inputs"] = torch.from_numpy(data["inputs"]).float()
        if data.get("targets") is not None:
            data["targets"] = data["targets"].long()
        else:
            data["targets"] = torch.from_numpy(data["lungsegment"]).long()

        return data


class SetLobeTarget:

    def __call__(self, data):
        data["targets"] = torch.from_numpy(data["lobe"])

        return data
