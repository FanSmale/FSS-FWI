import os
import re
import numpy as np

# Users do not need to run this .py file themselves, unless they want to add new training and testing data
# The purpose of this file is to generate a .txt list file for training and testing guidance.
# These .txt list files are located in the "./configuration" folder and have three suffixes: "_base", "_cont", "_lres".
# "_base" indicates that a training sample consists of {seismic records - velocity model}
# "_cont" indicates that a training sample consists of {seismic record - velocity model - velocity model contour}
# "_lres" indicates that a training sample consists of
# {seismic record - velocity model - inverted smoothed velocity model - velocity model contour}

if __name__ == '__main__':
    dataset_name = "CurveFaultA"

    id_boundary = {
        "FlatFaultA": [96, 108],
        "CurveFaultA": [96, 108],
        "CurveVelA": [48, 60]
    }

    ids = id_boundary[dataset_name]

    dir_path = r'.\{}'.format(dataset_name)

    file_ls = os.listdir(dir_path)
    file_ls.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    seismic_dir_list = []
    vmodel_dir_list = []
    cvmodel_dir_list = []
    lvmodel_dir_list = []

    for file in file_ls:
        if file[:6] == "vmodel":
            vmodel_dir_list.append(file)
        elif file[:7] == "seismic":
            seismic_dir_list.append(file)
        elif file[:7] == "cvmodel":
            cvmodel_dir_list.append(file)
        elif file[:7] == "lvmodel":
            lvmodel_dir_list.append(file)

    train_ids = list(np.arange(1, ids[0] + 1))
    test_ids = list(np.arange(ids[0] + 1, ids[1] + 1))

    # Only seismic data and velocity
    # print("train")
    # for pair in zip(seismic_dir_list, vmodel_dir_list):
    #     if int(re.findall(r'\d+', pair[0])[0]) in train_ids:
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[0], end=" ")
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[1])
    #
    # print("test")
    # for pair in zip(seismic_dir_list, vmodel_dir_list):
    #     if int(re.findall(r'\d+', pair[0])[0]) in test_ids:
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[0], end=" ")
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[1])

    # Add contour of velocity mdoel
    # print("train")
    # for pair in zip(seismic_dir_list, vmodel_dir_list, cvmodel_dir_list):
    #     if int(re.findall(r'\d+', pair[0])[0]) in train_ids:
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[0], end=" ")
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[1], end=" ")
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[2])
    #
    # print("test")
    # for pair in zip(seismic_dir_list, vmodel_dir_list, cvmodel_dir_list):
    #     if int(re.findall(r'\d+', pair[0])[0]) in test_ids:
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[0], end=" ")
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[1], end=" ")
    #         print(".\data\{}".format(dataset_name) + "\\" + pair[2])

    # Add contour and low resolution of velocity mdoel
    print("train")
    for pair in zip(seismic_dir_list, vmodel_dir_list, lvmodel_dir_list, cvmodel_dir_list):
        if int(re.findall(r'\d+', pair[0])[0]) in train_ids:
            print(".\data\{}".format(dataset_name) + "\\" + pair[0], end=" ")
            print(".\data\{}".format(dataset_name) + "\\" + pair[1], end=" ")
            print(".\data\{}".format(dataset_name) + "\\" + pair[2], end=" ")
            print(".\data\{}".format(dataset_name) + "\\" + pair[3])

    print("test")
    for pair in zip(seismic_dir_list, vmodel_dir_list, lvmodel_dir_list, cvmodel_dir_list):
        if int(re.findall(r'\d+', pair[0])[0]) in test_ids:
            print(".\data\{}".format(dataset_name) + "\\" + pair[0], end=" ")
            print(".\data\{}".format(dataset_name) + "\\" + pair[1], end=" ")
            print(".\data\{}".format(dataset_name) + "\\" + pair[2], end=" ")
            print(".\data\{}".format(dataset_name) + "\\" + pair[3])
