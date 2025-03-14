import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')


font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}
font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}


def minmax_normalize(img: np.ndarray, vmin: float, vmax: float):
    """
    normalization operation

    :param img:     input image
    :param vmin:    Minimum values in the image
    :param vmax:    Maximum values in the image
    :return:        Normalized image
    """

    img -= vmin
    img /= (vmax - vmin)
    return img


def minmax_denormalize(img: np.ndarray, vmin: float, vmax: float):
    """
    denormalization operation

    :param img:     input image
    :param vmin:    Minimum values in the image
    :param vmax:    Maximum values in the image
    :return:        Denormalized image
    """
    return img * (vmax - vmin) + vmin


def mute_direct_wave(common_shot_gather: np.ndarray, vmodel: np.ndarray):
    """
    mute the direct wave
    (Note that the velocity model must be the un-normalized version!)

    :param common_shot_gather:      Seismic records with direct waves
    :param vmodel:                  This seismic record corresponds to a velocity model (can be a low-frequency version)
    :return:                        Seismic records without direct waves
    """

    assert np.max(vmodel) != 1.0

    dx = 10
    nz, nx = vmodel.shape
    dt = 0.001
    muted_gather = common_shot_gather.copy()
    wave_duration = 82

    for index, shot_pos in enumerate([5, 15, 25, 35, 45, 55, 65]):

        mean_value = np.mean(muted_gather[index])

        x_array = np.arange(0, nx * dx, dx)
        v0 = vmodel[0, :]
        traveltimes = abs(np.cumsum(dx / v0) - np.cumsum(dx / v0)[shot_pos])
        for traceno in range(len(x_array)):
            muted_gather[index][0:int(traveltimes[traceno] / dt + wave_duration), traceno] = mean_value

    return muted_gather


def extract_contours(para_image: np.ndarray):
    """
    Use Canny to extract contour features

    :param para_image:  Velocity model (numpy)
    :return:            Binary contour structure of the velocity model (numpy)
    """

    image = para_image

    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image_to_255 = norm_image * 255
    norm_image_to_255 = norm_image_to_255.astype(np.uint8)
    canny = cv2.Canny(norm_image_to_255, 10, 15)
    bool_canny = np.clip(canny, 0, 1)
    return bool_canny


if __name__ == "__main__":
    pass
