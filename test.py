from parse_args import *
from metrics import Metrics
from fwi_dataset import FWIDataset
from net_control import NetworkControl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import minmax_denormalize
import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt


class Tester(object):
    """
    A series of network test
    """

    def __init__(self, para_args: argparse):
        """
        constructor

        :param para_args:   Object to store global variables
        """

        # Get global parameters
        self.args = para_args
        self.datasets = FWIDataset(self.args)
        self.datasets.load2memory()
        self.net_control = NetworkControl(self.datasets, self.args)

        # Initialize new values
        self.pre_vmodel = None

    def single_test(self, para_id: int):
        """
        Testing on a single sample

        :param para_id:     The position number of the data sample in the data file
        :return:
        """

        show_shot_id = 3
        minmax_info = self.datasets.maxmin_query[para_id]

        min_csg, max_csg, min_vel, max_vel = \
            minmax_info["csg_min"][show_shot_id], minmax_info["csg_max"][show_shot_id],\
            minmax_info["vel_min"], minmax_info["vel_max"]

        pre_vmodel, gt_vmodel = self.net_control.network_test(list(self.datasets[para_id]))
        metrics = Metrics()
        metrics.update_samples(gt_vmodel[0][0], pre_vmodel[0][0])
        metrics.print_metrics_multiline()

        self.pre_vmodel = minmax_denormalize(pre_vmodel[0][0],
                                             vmax=max_vel,
                                             vmin=min_vel)
        self.datasets[para_id][1][0] = minmax_denormalize(self.datasets[para_id][1][0],
                                                          vmax=max_vel,
                                                          vmin=min_vel)
        self.datasets[para_id][0][show_shot_id] = minmax_denormalize(self.datasets[para_id][0][show_shot_id],
                                                                     vmax=max_csg,
                                                                     vmin=min_csg)

        self.datasets.show_seismic(idx=para_id, shot_num=show_shot_id)
        self.datasets.show_vmodel(idx=para_id)
        self.pain_inversion_result()

    def multi_test(self):
        """
        Testing on multiple samples

        :return:
        """
        metrics = Metrics()

        for i, data_zip in enumerate(self.net_control.dataset_loader):
            pre_vmodels, gt_vmodels = self.net_control.network_test(data_zip)
            for k in range(self.net_control.batch_size):
                metrics.update_samples(gt_vmodels[k, 0, :, :], pre_vmodels[k, 0, :, :])
                metrics.print_metrics_singleline(i * self.net_control.batch_size + k + 1)

        metrics.avg_dict()
        result_dict = metrics.get_metrics()
        for metric in result_dict:
            print("The average of {}: {:.6f}".format(metric, result_dict[metric]))

    def save_lowres_inversion_results(self):
        """
        Saving LInvNet inversion results to a fixed location

        :return:
        """

        save_vm_unit = np.zeros([self.datasets.data_volume] + self.args.vmodel_shape).astype(np.float32)

        for index in range(self.datasets.data_size):
            pre_vmodel, gt_vmodel = self.net_control.network_test(list(self.datasets[index]))
            save_vm_unit[index % self.datasets.data_volume] = pre_vmodel[0]

            if (index + 1) % self.datasets.data_volume == 0:
                if self.datasets.is_training:
                    vmodel_id = (index + 1) // self.args.train_data_volume
                else:
                    vmodel_id = (index + 1) // self.args.test_data_volume + self.args.train_size // self.args.train_data_volume

                save_path = "L:/My Paper Code/double_res_FWI/data/{}/lvmodel{}.npy".format(self.datasets.dataset_name, vmodel_id)
                #save_path = "L:/My Paper Code/double_res_FWI/data/Marmousi2_Slice10970/lvmodel{}.npy".format(vmodel_id)
                print(save_path)
                np.save(save_path, save_vm_unit)

    def marmousi_test(self):

        from data.slice4marmousi2 import upsample_image, reconstruct_image, reconstruct_image_and_fill_missing_values

        pre_unit140 = np.zeros([self.datasets.data_volume] + self.args.vmodel_shape).astype(np.float32)
        gt_unit140 = np.zeros([self.datasets.data_volume] + self.args.vmodel_shape).astype(np.float32)
        crop_size = (140, 140)
        original_size = (2380, 1400)

        for index in range(self.datasets.data_size):

            pre_vmodel, gt_vmodel = self.net_control.network_test(list(self.datasets[index]))

            minmax_info = self.datasets.maxmin_query[index]

            min_vel, max_vel = minmax_info["vel_min"], minmax_info["vel_max"]

            gt_unit140[index, 0] = minmax_denormalize(gt_vmodel[0][0], vmax=max_vel, vmin=min_vel)
            pre_unit140[index, 0] = minmax_denormalize(pre_vmodel[0][0], vmax=max_vel, vmin=min_vel)

        gt_upsampled_images = [upsample_image(image, (140, 140)) for image in gt_unit140]
        gt_reconstructed_image = reconstruct_image(gt_upsampled_images, original_size, crop_size)
        # np.save("gt_Marmousi2.npy", gt_reconstructed_image)
        plt.imshow(gt_reconstructed_image)
        plt.show()

        pre_upsampled_images = [upsample_image(image, (140, 140)) for image in pre_unit140]
        pre_reconstructed_image = reconstruct_image(pre_upsampled_images, original_size, crop_size)
        # np.save("pr_{}_Marmousi2.npy".format(self.args.network_name), pre_reconstructed_image)
        plt.imshow(pre_reconstructed_image)
        plt.show()

    def pain_inversion_result(self, min_velo: int = None, max_velo: int = None):
        """
        Display the velocity model for inverse prediction within the current Tester class

        :param min_velo:    Minimum velocity values in the velocity model
        :param max_velo:    Maximum velocity values in the velocity model
        :return:
        """

        assert self.pre_vmodel is not None, "No predictions made yet."

        vmodel = self.pre_vmodel

        fig, ax = plt.subplots(figsize=(5.8, 6), dpi=150)

        vmin = min_velo if min_velo is not None else np.min(vmodel)
        vmax = max_velo if max_velo is not None else np.max(vmodel)

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


if __name__ == '__main__':

    ###############################################################
    # Some commands commonly used for testing different networks #
    ###############################################################

    # Testing for InversionNet
    # -n {} -net InversionNet -nep .\models\{}Model\InversionNet_120of120.pkl

    # Testing for VelocityGAN
    # -n {} -net VelocityGAN -nep .\models\{}Model\VelocityGAN_480of480.pkl

    # Testing for SeisDeepNET70
    # -n {} -net SeisDeepNET70 -nep .\models\{}Model\SeisDeepNET70_200of200.pkl

    # Testing for DDNet70
    # -n {} -net DDNet70 -nep .\models\{}Model\DDNet70_120of120.pkl

    # Testing for LInvNet
    # -n {} -net LInvNet -bv -nep .\models\{}Model\LInvNet_100of100.pkl

    # Testing for DenseInvNet
    # -n {} -net DenseInvNet -mc -nep .\models\{}Model\[CSPL+DW]DenseInvNet_120of120.pkl

    # main
    ds = "CurveVelA"
    batch_id = 0
    sample_id = 56
    multi_or_single = 0
    args_str0 = \
        "-n {} -net DenseInvNet -mc -nep .\models\{}Model\[CSPL+DW]DenseInvNet_120of120.pkl".format(ds, ds)
    args_str1 = args_str0 + " -r {}".format(batch_id)

    if multi_or_single == 0:
        args = parse_args(args_str0)
        temp_tester = Tester(args)
        temp_tester.multi_test()

        # # Save the inverted low-resolution velocity model to a fixed path.
        # # (Note that the inverted velocity model saved locally has been normalized.)
        # temp_tester.save_lowres_inversion_results()

        # # View the inversion results of all test samples in the Marmousi slice dataset.
        # temp_tester.marmousi_test()
    else:
        args = parse_args(args_str1)
        temp_tester = Tester(args)
        temp_tester.single_test(sample_id)

        # # Save inversion results for a single sample
        # np.save("pr_{}_{}_{}_{}.npy".format("DenseInvNet", ds, batch_id, sample_id), temp_tester.pre_vmodel)
