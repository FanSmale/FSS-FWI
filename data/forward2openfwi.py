from multiprocessing import Pool
import time
import numpy as np
import os
import re
import math


def id_single2batch(input: int, batch_size: int) -> (int, int):
    """

    :param input:       single id [1 ~ 54,000]
    :param batch_size:  batch size [500]
    :return:            (batch out_id, batch in_id)
    """
    return (input - 1) // batch_size + 1, (input - 1) % batch_size


def id_batch2single(out_id: int, in_id: int, batch_size: int) -> int:
    """

    :param out_id:      batch out_id  [1 ~ 108]
    :param in_id:       batch in_id [0 ~ 499]
    :param batch_size:  batch size [500]
    :return:            single id
    """
    return batch_size * (out_id - 1) + in_id + 1


def index_2dim(matrix, height_index_array, width_index_array):
    return matrix[height_index_array[:, np.newaxis], width_index_array]


def FD2D_elastic_oneshot(vmodel, vmodel_id, sz, sx, h, dx, dz, nt, dt, nodr, f0):
    """

    :param vmodel:      Velocity model
    :param vmodel_id:   ID of Velocity model
    :param sz:          The depth of seismic source
    :param sx:          The horizontal position of the seismic source
    :param h:           PML width
    :param dx:          x-axis interval
    :param dz:          z-axis interval
    :param nt:          Number of time samples
    :param dt:          Sampling time
    :param nodr:        Half the spatial difference order
    :param f0:          Source frequency
    :return:            single-shot seismic record
    """

    shot_time = time.time()

    nz, nx = vmodel.shape
    wave_source_ampl = 1
    Nz, Nx = nz + 2 * h, nx + 2 * h

    # Coefficients in differential calculation
    B = np.zeros([nodr, 1])
    B[0][0] = 1
    A = np.zeros([nodr, nodr])
    for i in range(nodr):
        A[i, :] = np.power(np.arange(1, 2 * nodr, 2), 2 * i + 1)
    C = np.dot(np.linalg.inv(A), B).reshape(-1)

    # Compute density matrix
    rho = 1000 * np.ones([Nz, Nx])
    vmodel_pad = np.pad(vmodel, [h, h], 'edge')

    # Generate source wave
    t = dt * np.arange(1, nt + 1, 1)
    t0 = 1 / f0
    source_wav = (1 - 2 * np.power((math.pi * f0 * (t - t0)), 2)) * np.exp(-np.power((math.pi * f0 * (t - t0)), 2))

    # Calculation of absorption coefficient of PML layer
    v_max = np.max(vmodel_pad)
    dp_z = np.zeros([Nz, Nx])
    dp_x = np.zeros([Nz, Nx])

    # Set up the upper and lower layers
    dp0_z = 3 * v_max / dz * (8 / 15 - 3 * h / 100 + 1 * h ** 2 / 1500)
    # Calculate absorption factors for edges
    dp_z[:h, :] = np.dot(dp0_z * np.power(np.arange(h, 0, -1) / h, 2).reshape(-1, 1), np.ones([1, Nx]))
    dp_z[(Nz - h):, :] = dp_z[h - 1::-1, :]

    # Set up left and right layers
    dp0_x = 3 * v_max / dx * (8 / 15 - 3 * h / 100 + 1 * h ** 2 / 1500)
    # Calculate absorption factors for edges
    dp_x[:, :h] = np.dot(np.ones([Nz, 1]), dp0_x * np.power(np.arange(h, 0, -1) / h, 2).reshape(1, -1))
    dp_x[:, (Nx - h):] = dp_x[:, (h - 1)::-1]

    # The elastic coefficient calculated based on generalized Hooke's law
    rho1 = rho.copy()
    rho2 = rho.copy()
    # Coeffi1 and Coeffi2 are the coefficients of the PML absorption factor along the x- and z-axis directions.
    Coeffi1 = (2 - dt * dp_x) / (2 + dt * dp_x)
    Coeffi2 = (2 - dt * dp_z) / (2 + dt * dp_z)
    # Coeffi3 and Coeffi4 are coefficients related to density (rho) and spatial step size (dx and dz),
    # used to account for the spatial derivative terms in the wavefield update equation.
    Coeffi3 = 1 / rho1 / dx * (2 * dt / (2 + dt * dp_x))
    Coeffi4 = 1 / rho2 / dz * (2 * dt / (2 + dt * dp_z))
    # Coeffi5 and Coeffi6 are coefficients related to the square of the density (rho) and velocity (vp) and the spatial
    # step size (dx and dz), used to account for the velocity and stress terms in the wavefield update equation.
    Coeffi5 = rho * np.power(vmodel_pad, 2) / dx * (2 * dt / (2 + dt * dp_x))
    Coeffi6 = rho * np.power(vmodel_pad, 2) / dz * (2 * dt / (2 + dt * dp_z))

    # Initialize before iteration
    # Set external space: all wavefield values are empty to prevent out-of-bounds
    NZ = Nz + 2 * nodr
    NX = Nx + 2 * nodr
    # The effective index area after applying the PML layer
    Znodes = np.arange(nodr, NZ - nodr, 1)
    Xnodes = np.arange(nodr, NX - nodr, 1)
    # Effective index area of original image
    znodes = np.arange(nodr + h, nodr + h + nz)
    xnodes = np.arange(nodr + h, nodr + h + nx)
    # Set the PML layer and the source location of the external space
    sz_pad = nodr + h + sz
    sx_pad = nodr + h + sx
    # Initialize the stress matrix and the matrix related to the velocity directional component
    Ut = np.zeros([NZ, NX])
    Uz = np.zeros([NZ, NX])
    Ux = np.zeros([NZ, NX])
    Vz = np.zeros([NZ, NX])
    Vx = np.zeros([NZ, NX])
    U = -1 * np.ones([nz, nx, nt])
    Psum = -1 * np.ones([Nz, Nx])

    for cur_time in range(nt):
        Ux[sz_pad, sx_pad] = Ux[sz_pad, sx_pad] + wave_source_ampl * source_wav[cur_time] / 2
        Uz[sz_pad, sx_pad] = Uz[sz_pad, sx_pad] + wave_source_ampl * source_wav[cur_time] / 2
        Ut = Ux + Uz  # Ut is the combination of the two component matrices Ux and Uz
        U[:, :, cur_time] = index_2dim(Ut, znodes, xnodes)

        Psum[:, :] = 0
        for i in range(1, nodr + 1):
            Psum = Psum + C[i - 1] * (index_2dim(Ut, Znodes, Xnodes + i) - index_2dim(Ut, Znodes, Xnodes + 1 - i))
        Vx[nodr:NZ - nodr, nodr:NX - nodr] = Coeffi1 * index_2dim(Vx, Znodes, Xnodes) - Coeffi3 * Psum

        Psum[:, :] = 0
        for i in range(1, nodr + 1):
            Psum = Psum + C[i - 1] * (index_2dim(Ut, Znodes + i, Xnodes) - index_2dim(Ut, Znodes + 1 - i, Xnodes))
        Vz[nodr:NZ - nodr, nodr:NX - nodr] = Coeffi2 * index_2dim(Vz, Znodes, Xnodes) - Coeffi4 * Psum

        Psum[:, :] = 0
        for i in range(1, nodr + 1):
            Psum = Psum + C[i - 1] * (index_2dim(Vx, Znodes, Xnodes - 1 + i) - index_2dim(Vx, Znodes, Xnodes - i))
        Ux[nodr:NZ - nodr, nodr:NX - nodr] = Coeffi1 * index_2dim(Ux, Znodes, Xnodes) - Coeffi5 * Psum

        Psum[:, :] = 0
        for i in range(1, nodr + 1):
            Psum = Psum + C[i - 1] * (index_2dim(Vz, Znodes - 1 + i, Xnodes) - index_2dim(Vz, Znodes - i, Xnodes))
        Uz[nodr:NZ - nodr, nodr:NX - nodr] = Coeffi2 * index_2dim(Uz, Znodes, Xnodes) - Coeffi6 * Psum

    common_shot_gather = U[1, :, :].T
    time_elapsed = time.time() - shot_time
    print("[vmodel{}.mat : Complete No. {} Shot] Forward Simulation for one shot completed! "
          "It takes {:.0f}m {:.0f}s".format(vmodel_id, sx, time_elapsed // 60, time_elapsed % 60))
    return common_shot_gather


def forward2all(start_id: int = 1, read_path: str = ".\CurveVelA", pack_size: int = 500):
    """
    :param start_id:        The index of the first velocity model file need to forward modeling
    :param read_path:       The path to read
    :param pack_size:       Packed file size (default is 500)

    Perform forward modeling for each velocity model to obtain seismic records
    Note that a single velocity model corresponds to one .npy seismic record file.
    :return:
    """

    path = read_path
    output_dir = r".\temporary_storage"
    pro_pool_set = [7]

    file_ls = os.listdir(path)
    file_ls.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    # file_ls = sorted([f for f in os.listdir(file_ls) if f.startswith('single_seismic') and f.endswith('.npy')])

    vmodel_dir_list = []

    for file in file_ls:
        if file[:6] == "vmodel":
            vmodel_dir_list.append(path + "\\" + file)

    start_outid, start_inid = id_single2batch(start_id, pack_size)

    for vmodel_dir in vmodel_dir_list:

        cur_outid = int(re.findall(r'\d+', vmodel_dir.split("\\")[-1])[0])

        if cur_outid < start_outid:
            continue
        elif cur_outid == start_outid:
            cur_inid = start_inid
        else:
            cur_inid = 0

        multi_vmodel = np.load(vmodel_dir).astype(np.float64)
        print("------------- Read velocity model: {} -------------".format(vmodel_dir))

        while cur_inid < pack_size:

            vmodel_time = time.time()

            vmodel = multi_vmodel[cur_inid][0]

            vmodel_id = id_batch2single(cur_outid, cur_inid, pack_size)

            CSG = np.zeros([1, 7, 1000, 70], dtype=np.float32)
            process_list = []
            process_batch_list = pro_pool_set
            init_shot_position = 5

            for max_process_num in process_batch_list:
                pool = Pool(processes=max_process_num)  # Set process pool size
                # Set the shot positions for each forward simulation in the current process pool
                position_range = range(init_shot_position, init_shot_position + 10 * max_process_num, 10)

                # allocate the process
                for temp_source_x in position_range:
                    process_list.append(pool.apply_async(FD2D_elastic_oneshot, (vmodel, vmodel_id, 0, temp_source_x,
                                                                                10, 10, 10, 1000, 0.001, 3, 25)))
                pool.close()
                pool.join()

                # Receive results from each process
                for temp_source_x in position_range:
                    index = temp_source_x // 10
                    CSG[0][index] = process_list[index].get()

                init_shot_position += (max_process_num * 10)

            time_elapsed = time.time() - vmodel_time
            print("[{}) Finish!] runtime: {:.0f}m {:.0f}s".format(vmodel_id, time_elapsed // 60, time_elapsed % 60))
            np.save(output_dir + "\\" + "single_seismic{}.npy".format(vmodel_id), CSG.astype(np.float32))

            cur_inid += 1


def data_pack(dir: str = "...", keyword: str = "...", pack_size: int = 500, para_range: list = [1, 24000],
              save_dir: str = "...", save_keyword: str ="..."):
    """
    Pack multiple seismic record files (1x7x1000x70 -> 500x7x1000x70)

    :param dir:             The path of the file that needs to be packaged
    :param keyword:         Single seismic record file name
    :param pack_size:       Packed file size (default is 500)
    :param para_range:      Number range of packaged files (e.g., [0, 48000] is training set for CurveVelA)
    :param save_dir:        Packaged file storage path
    :param save_keyword:    Packaged seismic record file name
    :return:
    """
    file_ls = os.listdir(dir)
    file_ls.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    dir_list = []

    for file in file_ls:
        if file[:len(keyword)] == keyword:
            dir_list.append(dir + "\\" + file)

    pack_num = int((para_range[1] - para_range[0] + 1) / pack_size)
    print("There are a total of {} files, and we will package {} of them.".format(len(dir_list),
                                                                                  para_range[1] - para_range[0] + 1))
    print("We will pack {} copies.".format(pack_num))

    temp = pack_num
    while pack_num > 0:

        data_zip = []
        for i in range(pack_size):
            print("load: {}".format(dir_list[0]))
            data_zip.append(np.load(dir_list[0]))
            dir_list = dir_list[1:]
        data_zip = np.vstack(data_zip)

        save_id = temp - pack_num + 1
        file_save_dir = save_dir + "\\" + save_keyword + str(save_id) + ".npy"
        print("save: " + file_save_dir)
        np.save(file_save_dir, data_zip.astype(np.float32))

        pack_num -= 1


if __name__ == '__main__':
    # Please use 'forward2all' to forward modeling all velocity models one by one to generate seismic records,
    # and store them in the folder 'temporary_storage'.
    # Then use 'data_pack' to pack the seismic records in this folder according to the specified 'pack_size'.

    info_dict = {
        "CurveVelA_training_set": {
            "pack_size": 500,
            "para_range": [1, 24000],
            "save_dir": ".\CurveVelA"
        },
        "CurveVelA_testing_set": {
            "pack_size": 500,
            "para_range": [1, 24000],
            "save_dir": ".\CurveVelA"
        },
        "CurveFaultA_training_set": {
            "pack_size": 500,
            "para_range": [1, 48000],
            "save_dir": ".\CurveFaultA"
        },
        "CurveFaultA_testing_set": {
            "pack_size": 500,
            "para_range": [48001, 54000],
            "save_dir": ".\CurveFaultA"
        },
        "FlatFaultA_training_set": {
            "pack_size": 500,
            "para_range": [1, 48000],
            "save_dir": ".\FlatFaultA"
        },
        "FlatFaultA_testing_set": {
            "pack_size": 500,
            "para_range": [48001, 54000],
            "save_dir": ".\FlatFaultA"
        },
        "Marmousi2_training_set": {
            "pack_size": 400,
            "para_range": [1, 10800],
            "save_dir": ".\Marmousi2_Slice10970"
        },
        "Marmousi2_testing_set": {
            "pack_size": 170,
            "para_range": [10801, 10970],
            "save_dir": ".\Marmousi2_Slice10970"
        }
    }

    name = "CurveVelA_training_set"

    forward2all(start_id=info_dict[name]["para_range"][0],
                read_path=info_dict[name]["save_dir"],
                pack_size=info_dict[name]["pack_size"])

    data_pack(dir=r".\temporary_storage",
              keyword="single_seismic",
              pack_size=info_dict[name]["pack_size"],
              para_range=info_dict[name]["para_range"],
              save_dir=info_dict[name]["save_dir"],
              save_keyword="seismic")
