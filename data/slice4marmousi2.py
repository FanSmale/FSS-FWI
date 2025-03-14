from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import torch
import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt


font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}

font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}


def load_marmousi2(data_dir, num_dims, vmodel_dim):
    """
        Load the true model
    """

    if num_dims != len(vmodel_dim.reshape(-1)):
        raise Exception('Please check the size of model_true!!')
    if num_dims == 2:
        model_true = (np.fromfile(data_dir, np.float32).reshape(vmodel_dim[1], vmodel_dim[0]))
        model_true = np.transpose(model_true, (1, 0))  # I prefer having depth direction first
    else:
        raise Exception('Please check the size of model_true!!')

    model_true = torch.Tensor(model_true)

    return model_true


def pain_marmousi_velocity_model(para_velocity_model, min_velocity, max_velocity, is_colorbar = 1):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_velocity_model: Velocity model (70 x 70) (numpy)
    :param min_velocity:        Upper limit of velocity in the velocity model
    :param max_velocity:        Lower limit of velocity in the velocity model
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    :return:
    '''

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=(7, 3), dpi=150)

    im = ax.imshow(para_velocity_model, extent=[0, 17, 3.5, 0], vmin=min_velocity, vmax=max_velocity)

    ax.set_xlabel('Position (km)', font18)
    ax.set_ylabel('Depth (km)', font18)
    ax.set_xticks(range(0, 18, 2))
    ax.set_xticks([17], minor=True)
    ax.set_yticks(np.linspace(0, 3.5, 8))
    ax.set_xticklabels(labels=[' ', 2, 4, 6, 8, 10, 12, 14, 16], size=12)
    ax.set_yticklabels(labels=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], size=12)

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.95)
    else:
        plt.rcParams['font.size'] = 14      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.35)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(min_velocity, max_velocity, 7), format=mpl.ticker.StrMethodFormatter('{x:.0f}'))
        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)

    plt.show()


def sliding_window_crop(matrix, window_size, step_size):
    """
    Sliding window cropping of the matrix

    :param matrix:          Input matrix
    :param window_size:     Sliding window size
    :param step_size:       Sliding stride
    :return:                List of cropped images
    """
    h, w = matrix.shape[:2]
    crops = []
    for y in range(0, h - window_size[0] + 1, step_size[0]):
        for x in range(0, w - window_size[1] + 1, step_size[1]):
            crop = matrix[y:y + window_size[0], x:x + window_size[1]]
            crops.append(crop)
    return crops


def grid_crop(matrix, crop_size):
    """
    Grid clipping of the matrix

    :param matrix:          Input matrix
    :param crop_size:       Cropping size
    :return:                List of cropped images
    """
    h, w = matrix.shape[:2]
    crops = []
    num_h = h // crop_size[0]
    num_w = w // crop_size[1]
    for i in range(num_h):
        for j in range(num_w):
            y = i * crop_size[0]
            x = j * crop_size[1]
            crop = matrix[y:y + crop_size[0], x:x + crop_size[1]]
            crops.append(crop)
    return crops


def downsample_image(image, target_size):
    """
    Downsample

    :param image:           Input image
    :param target_size:     Target size
    :return:                Downsampled image
    """
    h, w = target_size
    return image[::2, ::2][:h, :w]


def upsample_image(image, target_size):
    """
    Upsample

    :param image:           Input image
    :param target_size:     Target size
    :return:                Upsampled image
    """
    h, w = target_size
    upsampled = np.zeros(target_size, dtype=image.dtype)
    upsampled[::2, ::2] = image[0, :h, :w]
    upsampled[1::2, ::2] = upsampled[::2, ::2]
    upsampled[::2, 1::2] = upsampled[::2, ::2]
    upsampled[1::2, 1::2] = upsampled[::2, ::2]
    return upsampled


def reconstruct_image(cropped_images, original_size, crop_size):
    """
    Reconstruct the original image from the cropped image (faster)

    :param cropped_images:  List of cropped images
    :param original_size:   The size of original image
    :param crop_size:       Cropping size
    :return:                Reconstructed original image
    """
    num_h = original_size[0] // crop_size[0]
    num_w = original_size[1] // crop_size[1]
    reconstructed = np.zeros(original_size, dtype=cropped_images[0].dtype)
    index = 0
    for i in range(num_h):
        for j in range(num_w):
            y = i * crop_size[0]
            x = j * crop_size[1]
            reconstructed[y:y + crop_size[0], x:x + crop_size[1]] = cropped_images[index]
            index += 1
    return reconstructed


def concatenate_horizontal(mat1, mat2, smooth_width=10):
    """
    Horizontal spaced stitching
    :param mat1:            First matrix
    :param mat2:            Second matrix
    :param smooth_width:    Interval width
    :return:                The spliced matrix
    """
    h1, w1 = mat1.shape
    h2, w2 = mat2.shape
    assert h1 == h2, "Heights of the two matrices must be equal"
    result = np.zeros((h1, w1 + w2))
    result[:, :w1 - smooth_width] = mat1[:, :w1 - smooth_width]
    result[:, w1 + smooth_width:] = mat2[:, smooth_width:]
    for i in range(smooth_width):
        weight = (i + 1) / (smooth_width + 1)
        result[:, w1 - smooth_width + i] = (1 - weight) * mat1[:, w1 - smooth_width + i] + weight * mat2[:, i]
    return result


def concatenate_vertical(mat1, mat2, smooth_width=10):
    """
    Vertical spaced stitching
    :param mat1:            First matrix
    :param mat2:            Second matrix
    :param smooth_width:    Interval width
    :return:                The spliced matrix
    """
    h1, w1 = mat1.shape
    h2, w2 = mat2.shape
    assert w1 == w2, "Widths of the two matrices must be equal"
    result = np.zeros((h1 + h2, w1))
    result[:h1 - smooth_width, :] = mat1[:h1 - smooth_width, :]
    result[h1 + smooth_width:, :] = mat2[smooth_width:, :]
    for i in range(smooth_width):
        weight = (i + 1) / (smooth_width + 1)
        result[h1 - smooth_width + i, :] = (1 - weight) * mat1[h1 - smooth_width + i, :] + weight * mat2[i, :]
    return result


def fill_missing_values(image: np.ndarray, visualize: bool = False):
    """
    Fill missing values ​​(grid locations with value 0) in an image using 2D interpolation

    :param image:           Input image (2D)
    :param visualize:       Whether to display the comparison image before and after interpolation
    :return:                The interpolated image
    """

    if len(image.shape) == 3:
        image = image[:, :, 0]
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    non_zero_mask = image > 0
    points = np.column_stack((x[non_zero_mask], y[non_zero_mask]))
    values = image[non_zero_mask]

    grid_z = griddata(points, values, (x, y), method='linear')

    filled_image = np.nan_to_num(grid_z)

    if visualize:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title("Original Image with Missing Values")
        plt.imshow(image, cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Image After Interpolation")
        plt.imshow(filled_image, cmap='viridis')
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    return filled_image


def reconstruct_image_and_fill_missing_values(cropped_images, original_size, crop_size, smooth_width=10):
    """
    Reconstruct the original image from the cropped image
    while ensuring the pasted surface is as smooth as possible (slower)

    :param cropped_images:  List of cropped images
    :param original_size:   The size of original image
    :param crop_size:       Cropping size
    :return:                Reconstructed original image
    """
    num_h = original_size[0] // crop_size[0]
    num_w = original_size[1] // crop_size[1]
    reconstructed = np.zeros(original_size, dtype=cropped_images[0].dtype)
    index = 0
    for i in range(num_h):
        row = None
        for j in range(num_w):
            y = i * crop_size[0]
            x = j * crop_size[1]
            if j == 0:
                row = cropped_images[index]
            else:
                row = concatenate_horizontal(row, cropped_images[index], smooth_width)
            index += 1
        if i == 0:
            reconstructed[0:row.shape[0], 0:row.shape[1]] = row
        else:
            reconstructed[:(i + 1) * crop_size[1], :] = concatenate_vertical(reconstructed[:i * crop_size[1], :],
                                                                                    row, smooth_width)
    return fill_missing_values(reconstructed)


if __name__ == '__main__':
    window_size = (140, 140)
    crop_size = (140, 140)
    step_size = (50, 50)
    nz = 2801               # Remove the water layer, the height is 2381
    ny = 13601

    #######################
    # Clipping  reasoning #
    #######################
    # temp_nz = nz - 420      # Remove the water layer
    # w_size = crop_size[0]
    # stride = step_size[0]
    #
    # # 6000 (train) + 1400 (test) + 6201 (train) = 13601
    # ny_1 = 6000
    # ny_2 = 6201
    # ny_3 = 1400
    #
    # h_num = (temp_nz - w_size) // stride + 1
    # w_num_1 = (ny_1 - w_size) // stride + 1
    # w_num_2 = (ny_2 - w_size) // stride + 1
    #
    # # Wasted pixels
    # # print((temp_nz - w_size) % stride)
    # # print((ny_1 - w_size) % stride)
    # # print((ny_2 - w_size) % stride)
    #
    # print("The num of training set:", h_num * (w_num_1 + w_num_2))
    # print("The num of testing set:", (temp_nz // crop_size[0]) * (ny_3 // crop_size[1]))

    #######################
    # 1. Show Marmousi II #
    #######################

    data_dir = ''
    data_name = 'MODEL_P-WAVE_VELOCITY_1.25m.bin'  # 145 MB
    num_dims = 2

    vmodel_dim = np.array([nz, ny])
    data_path = data_dir + data_name
    model_true = load_marmousi2(data_path, num_dims, vmodel_dim).cpu().numpy()  # [2801, 13601]
    model_true = model_true[420:, :]  # Remove the water layer [2381, 13601]
    vmin, vmax = np.min(model_true), np.max(model_true)
    pain_marmousi_velocity_model(model_true, vmin, vmax)

    train_region1 = model_true[:, :6000]
    print(train_region1.shape)
    train_region2 = model_true[:, 6000 + 1400:]
    print(train_region2.shape)
    test_region = model_true[:, 6000: 6000 + 1400]
    print(test_region.shape)

    ############################
    # 2. Training set clipping #
    ############################

    # Note that after clipping there may be some velocity models that only have one velocity,
    # and these models need to be deleted and replaced with other models.
    # This operation can be implemented by the reader.

    # crops1 = sliding_window_crop(train_region1, window_size, step_size)
    # crops2 = sliding_window_crop(train_region2, window_size, step_size)
    # downsampled_crops1 = [downsample_image(crop, (70, 70)) for crop in crops1]
    # downsampled_crops2 = [downsample_image(crop, (70, 70)) for crop in crops2]
    # num_files = 10800 // 400
    # num_per_file = 400
    # all_crops = downsampled_crops1 + downsampled_crops2
    # for i in range(num_files):
    #     start = i * num_per_file
    #     end = (i + 1) * num_per_file
    #     file_name = f'./Marmousi2_Slice10970/vmodel{i + 1}.npy'
    #
    #     data = np.array(all_crops[start:end])
    #     data = data.reshape(len(data), 1, 70, 70)
    #     np.save(file_name, data)

    ############################
    # 2. Testing set clipping #
    ############################

    # # Grid clipping of the matrix
    # crops = grid_crop(test_region, crop_size)
    # downsampled_crops = [downsample_image(crop, (70, 70)) for crop in crops]
    #
    # data = np.array(downsampled_crops).reshape(len(downsampled_crops), 1, 70, 70)
    #
    # # Save directly as a single file
    # np.save('./Marmousi2_Slice10970/vmodel28.npy', np.array(data))

    ################
    # 3. Data view #
    ################
    # v = np.load(r".\Marmousi2_Slice10970\vmodel11.npy")
    # print(v.shape)
    # plt.imshow(v[399][0])
    # plt.show()

    ##############################
    # 4. Test set reconstruction #
    ##############################

    # downsampled_crops = np.load(r'.\Marmousi2_Slice10970\vmodel28.npy', allow_pickle=True)
    # original_size = (2380, 1400)
    # upsampled_images = [upsample_image(image, (140, 140)) for image in downsampled_crops]
    # reconstructed_image = reconstruct_image(upsampled_images, original_size, crop_size)
    # plt.imshow(reconstructed_image)
    # plt.show()
