import cv2
import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt

from parse_args import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import Dataset


class FWIDataset(Dataset):
    """
    A series of operations on data
    """
    def __init__(self, para_args: argparse):
        """
        constructor

        :param para_args:   The object to store global variables
        """

        args = para_args

        # GGet global parameters
        self.dataset_name = args.dataset_name
        self.is_training = args.is_training

        self.shot_num = args.shot_num
        self.data_size = args.train_size if self.is_training else args.test_size
        self.data_volume = args.train_data_volume if self.is_training else args.test_data_volume
        self.readtxt_dir = args.readtxt_dir
        self.vmodel_shape = args.vmodel_shape
        self.seismic_shape = args.seismic_shape
        self.maxmin_query = []

        # Determine which data we need to read and where it is located
        self.data_zip = []
        self.data_order = "sv"
        self.data_path_lst = self.determine_where2read(para_args.read_range)
        self.dir_num = len(self.data_path_lst[0].split("+"))

    def determine_where2read(self, para_read_range):
        """
        Obtain the path to the data that needs to be read

        :param para_read_range:     Data range of reading.
        :return:                    List of stored path strings.
        """

        temp_list = []
        read_range = para_read_range
        with open(self.readtxt_dir, 'r') as file:
            for line in file:
                temp_list.append(line.strip())
        self.data_order = temp_list[0]
        if self.is_training:
            temp_list = temp_list[temp_list.index("train") + 1: temp_list.index("test")]
        else:
            temp_list = temp_list[temp_list.index("test") + 1:]

        if isinstance(read_range, list) and len(read_range) == 1:
            return temp_list[read_range[0]: read_range[0] + 1]
        elif isinstance(read_range, list):
            return temp_list[read_range[0]: read_range[1]]
        else:
            return temp_list

    def load2memory(self):
        """
        After determining which and where data to read, start reading.
        This operation may occupy a large amount of memory.
        """

        assert self.data_path_lst != [], 'Variable "data_path_lst" cannot be empty'
        for i in range(self.dir_num):
            self.data_zip.append([])
        for path_zip in self.data_path_lst:
            for idx, path in enumerate(path_zip.split("+")):
                print(path)
                self.data_zip[idx].append(np.load(path.rstrip('\n')).astype(np.float32))

    def storage_maxmin_value(self):
        """
        Used to store the maximum and minimum values of each data.
        (Note, only the first two items of data_zip are stored)
        It is convenient for querying during denormalization.
        :return:
        """

        for i in range(len(self)):
            temp_dict = {
                "csg_min": [],
                "csg_max": [],
                "vel_min": 0.0,
                "vel_max": 0.0
            }
            batch_idx, sample_idx = i // self.data_volume, i % self.data_volume
            temp_dict["vel_min"] = np.min(self.data_zip[1][batch_idx][sample_idx])
            temp_dict["vel_max"] = np.max(self.data_zip[1][batch_idx][sample_idx])
            for shot in range(self.shot_num):
                temp_dict["csg_min"].append(np.min(self.data_zip[0][batch_idx][sample_idx][shot]))
                temp_dict["csg_max"].append(np.max(self.data_zip[0][batch_idx][sample_idx][shot]))
            self.maxmin_query.append(temp_dict)

    def __getitem__(self, idx: int = 0):
        """
        :param idx:     a value range from 0 to "data_size"
        :return:        a tuple of data
        """
        batch_idx, sample_idx = idx // self.data_volume, idx % self.data_volume
        temp_data_zip = []
        for i in range(self.dir_num):
            temp_data_zip.append(self.data_zip[i][batch_idx][sample_idx])

        return tuple(temp_data_zip)

    def __len__(self):
        """
        :return: Number of data and data files
        """
        return self.data_volume * len(self.data_path_lst)

    def show_vmodel(self, idx: int = 0, min_velo: int = None, max_velo: int = None):
        """
        Display velocity model within the current FWIDataset class

        :param idx:         The position of the displayed data in the entire read dataset
        :param min_velo:    Minimum velocity values in the velocity model
        :param max_velo:    Maximum velocity values in the velocity model
        :return:
        """

        assert self.data_zip != [], "Please read the data into memory before presenting the data"

        batch_idx, sample_idx = idx // self.data_volume, idx % self.data_volume

        vmodel = self.data_zip[1][batch_idx][sample_idx][0]

        vmin = min_velo if min_velo is not None else np.min(vmodel)
        vmax = max_velo if max_velo is not None else np.max(vmodel)

        fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)

        im = ax.imshow(vmodel, extent=[0, 0.7, 0.7, 0], vmin=vmin, vmax=vmax)

        font18 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}

        ax.set_xlabel('Position (km)', font18)
        ax.set_ylabel('Depth (km)', font18)
        ax.set_xticks(np.linspace(0, 0.7, 8))
        ax.set_yticks(np.linspace(0, 0.7, 8))
        ax.set_xticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)
        ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], size=18)

        plt.rcParams['font.size'] = 14
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.35)

        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal',
                     ticks=np.linspace(vmin, vmax, 7), format=mpl.ticker.StrMethodFormatter('{x:.0f}'))

        plt.subplots_adjust(bottom=0.10, top=0.95, left=0.13, right=0.95)
        plt.show()

    def show_seismic(self, shot_num: int = 0, idx: int = 0):
        """
        Display seismic records within the current FWIDataset class

        :param shot_num:        Display the shot number in the seismic records
        :param idx:             The position of the displayed data in the entire read dataset
        :return:
        """

        assert self.data_zip != [], "Please read the data into memory before presenting the data"

        batch_idx, sample_idx = idx // self.data_volume, idx % self.data_volume
        seismic = self.data_zip[0][batch_idx][sample_idx][shot_num]

        data = cv2.resize(seismic, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)
        fig, ax = plt.subplots(figsize=(6.1, 8.2), dpi=120)

        # Display seismic records with direct waves
        im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-0.65, vmax=0.75)
        # Display seismic records without direct waves
        # im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-0.045, vmax=0.045)

        font21 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 21}

        ax.set_xlabel('Position (km)', font21)
        ax.set_ylabel('Time (s)', font21)
        ax.set_xticks(np.linspace(0, 0.7, 5))
        ax.set_yticks(np.linspace(0, 1.0, 5))
        ax.set_xticklabels(labels=[0, 0.17, 0.35, 0.52, 0.7], size=21)
        ax.set_yticklabels(labels=[0, 0.25, 0.5, 0.75, 1.0], size=21)

        plt.rcParams['font.size'] = 14  # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.3)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')
        plt.subplots_adjust(bottom=0.09, top=0.98, left=0.15, right=0.99)
        plt.show()


def example_test():
    """
    An Example of using the FWIDataset class

    :return:
    """
    # Test: Display the 234-th data pair in the 3rd testing file (Only 1 file was read into memory)
    temp_dataset = FWIDataset(parse_args("-n CurveVelA -r 2"))
    temp_dataset.load2memory()
    temp_data_zip = temp_dataset[233]
    temp_dataset.show_seismic(3, idx=233)  # If the idx is greater than 499, it crosses the boundary
    temp_dataset.show_vmodel(idx=233)

    seismic = temp_data_zip[0]
    print(seismic.shape)

    vmodel = temp_data_zip[1]
    print(vmodel.shape)

    # Test: Display the 234-th data pair in the 2nd testing file (12 files were read into memory)
    # temp_dataset = FWIDataset(parse_args("-n CurveVelA"))
    # temp_dataset.load2memory()
    # temp_dataset.show_seismic(3, idx=733)       # If the idx is greater than 5999, it crosses the boundary
    # print(temp_dataset[733][0].shape)
    # temp_dataset.show_vmodel(idx=733)
    # print(temp_dataset[733][1].shape)

    # Test: Read the 3rd to 5th files and display the 250-th file in order (3 files were read into memory)
    # temp_dataset = FWIDataset(parse_args("-n CurveVelA -r 2 6"))
    # temp_dataset.load2memory()
    # print(temp_dataset[250][0].shape)
    # temp_dataset.show_seismic(3, idx=250)
    # print(temp_dataset[250][1].shape)
    # temp_dataset.show_vmodel(idx=250)

    # Maximum Memory Data Read Test
    # temp_dataset = FWIDataset(parse_args("-n CurveFaultA -T"))
    # temp_dataset.load2memory()
    # temp_dataset.show_vmodel(idx=233)


if __name__ == '__main__':
    example_test()

