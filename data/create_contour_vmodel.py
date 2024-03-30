import os
import re
import numpy as np

from utils import extract_contours

# If you want to train DDNet70 and DenseInvNet, running this .py file is necessary.
# This .py file converts the velocity model into a contour image of the velocity model.
# Note that the normal velocity model name must be vmodelXX.npy.
# At the same time, the generated contour file is cvmodelXX.npy.

if __name__ == '__main__':

    dir_path = r'.\CurveVelA'

    file_ls = os.listdir(dir_path)
    file_ls.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    vmodel_dir_list = []

    for file in file_ls:
        if file[:6] == "vmodel":
            vmodel_dir_list.append(file)

    for filename in vmodel_dir_list:
        strid = re.findall(r'\d+', filename)[0]
        path = dir_path + "\\" + filename
        print(path, end="--(create)-->")

        temp_vmodel = np.load(path)
        for i in range(temp_vmodel.shape[0]):
            temp_vmodel[i][0] = extract_contours(temp_vmodel[i][0])
        output_path = dir_path + "\\" + "cvmodel{}.npy".format(strid)
        print(output_path)

        np.save(output_path, temp_vmodel)
