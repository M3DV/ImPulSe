from copy import deepcopy

import SimpleITK as sitk



INTERPOLATIONS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
}




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


def resample_to_shape(arr, target_shape, interpolation):
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


