import torch
import torch.nn as nn
import utils
import numpy as np
import skimage.filters
import lpips
import math

from parse_args import *
from torch.utils.data import RandomSampler, DataLoader
from fwi_dataset import FWIDataset

from networks.InversionNet import InversionNet
from networks.VelocityGAN import VelocityGAN, Discriminator, WassersteinGP, GeneratorLoss
from networks.DDNet70 import DDNet70Model, LossDDNet
from networks.LInvNet import LInvNet
from networks.DenseInvNet import DenseInvNet, MixLoss
from networks.SeisDeepNET70 import SeisDeepNET70


class NetworkControl(object):
    """
    A series of operations for the network
    """
    def __init__(self, para_fwi_dataset: FWIDataset, para_args: argparse):
        """
        Constructor

        :param para_fwi_dataset:    The object to store data
        :param para_args:           The object to store global variables
        """
        # Get global parameters
        self.args = para_args
        self.net_model = None
        self.optimizer = None
        self.net_model = None
        self.optimizer_D = None
        self.criterion = None
        self.criterion_D = None
        self.fwi_dataset = para_fwi_dataset
        self.network_name = self.args.network_name
        self.network_path = self.args.network_path
        self.batch_size = self.args.train_batch_size if self.fwi_dataset.is_training else self.args.test_batch_size
        self.in_channel = self.args.in_channel
        self.is_use_cuda = self.args.is_use_cuda
        self.is_norm_csg = self.args.is_norm_csg
        self.is_norm_vms = self.args.is_norm_vms
        self.is_blur_vms = self.args.is_blur_vms
        self.is_mute_csg = self.args.is_mute_csg
        self.display_step = self.args.display_step
        self.lpips_object = None
        self.gauss_spl_func = None

        # Network-specific initialization
        if self.network_name == "InversionNet":
            self.net_model = InversionNet(self.in_channel)
            self.criterion = nn.MSELoss()
        elif self.network_name == "VelocityGAN":
            self.net_model = VelocityGAN(self.in_channel)
            self.net_model_D = Discriminator()
            self.criterion = GeneratorLoss(self.args.lambda_g1v, self.args.lambda_g2v, self.args.lambda_adv)
            self.criterion_D = WassersteinGP("cuda", self.args.lambda_gp)
            if torch.cuda.is_available() and self.is_use_cuda:
                self.net_model_D = torch.nn.DataParallel(self.net_model_D, device_ids=[0]).cuda()
            self.optimizer_D = torch.optim.Adam(self.net_model_D.parameters(), lr=self.args.lr_discriminator)
        elif self.network_name == "DDNet70":
            self.net_model = DDNet70Model(self.in_channel)
            self.criterion = LossDDNet(weights=self.args.loss_weight)
        elif self.network_name == "LInvNet":
            self.net_model = LInvNet(self.in_channel)
            self.criterion = nn.MSELoss()
        elif self.network_name == "SeisDeepNET70":
            self.net_model = SeisDeepNET70(self.in_channel)
            self.criterion = nn.MSELoss()
        elif self.network_name == "DenseInvNet":

            # Escape some information in Argparse object
            if isinstance(self.args.param_list[0], str):
                for i in range(len(self.args.param_list)):
                    self.args.param_list[i] = self.args.param_list[i].split("_")
                    for p in range(len(self.args.param_list[i])):
                        self.args.param_list[i][p] = eval(self.args.param_list[i][p])

            self.net_model = DenseInvNet(self.in_channel)
            self.criterion = MixLoss(weights=[self.args.param_list[self.args.cur_stage][0],
                                              self.args.param_list[self.args.cur_stage][1]])
            self.lpips_object = lpips.LPIPS(net='alex', version="0.1")
        else:
            print('The "model_keyword" parameter selected in the load_network(...) '
                  'is the undefined network model keyword! Please check!')
            exit(0)

        # Reading network and cuda initialization
        if self.network_path != "":
            self.net_model = self.network_read(self.network_path)
        if torch.cuda.is_available() and self.is_use_cuda:
            self.net_model = torch.nn.DataParallel(self.net_model, device_ids=[0]).cuda()
        self.optimizer = torch.optim.Adam(self.net_model.parameters(), lr=self.args.learning_rate)

        # Basic data processing
        self.data_preprocessing()
        self.dataset_loader = self.get_dataset_loader()

    def network_read(self, pkl_src_name: str) -> nn.Module:
        '''
        Read .pkl network file

        :param pkl_src_name:    Read Path
        :return:                network object
        '''

        print("Read .pkl: {}".format(pkl_src_name))
        model = torch.load(pkl_src_name)
        try:
            self.net_model.load_state_dict(model)
        except RuntimeError:
            print("This model is obtained by multi-GPU training...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model.items():
                name = k[7:]
                new_state_dict[name] = v
            self.net_model.load_state_dict(new_state_dict)
        return self.net_model

    def network_save(self, save_path: str):
        """
        Save the network locally as .pkl

        :param save_path:       Storage Path
        :return:
        """
        torch.save(self.net_model.state_dict(), save_path)
        print("Network saved.")

    def data_preprocessing(self):
        """
        Based on the definitions in Argparse object, perform a series of operations on different data
        (including normalization, Gaussian blur, etc.)

        :return:
        """
        common_shot_gathers = self.fwi_dataset.data_zip[0]
        velocity_models = self.fwi_dataset.data_zip[1]

        print("· Data preprocessing in progress...")

        # Elimination of direct waves from seismic records (generally used for DenseInvNet)
        if self.is_mute_csg:
            for i in range(len(common_shot_gathers)):
                for j in range(common_shot_gathers[i].shape[0]):
                    common_shot_gathers[i][j] = utils.mute_direct_wave(common_shot_gathers[i][j],
                                                                       velocity_models[i][j][0])
        self.fwi_dataset.storage_maxmin_value()

        # Normalized seismic record
        if self.is_norm_csg:
            for i in range(len(common_shot_gathers)):
                for j in range(common_shot_gathers[i].shape[0]):
                    for k in range(common_shot_gathers[i][j].shape[0]):
                        common_shot_gathers[i][j][k] = utils.minmax_normalize(common_shot_gathers[i][j][k],
                                                                              np.min(common_shot_gathers[i][j][k]),
                                                                              np.max(common_shot_gathers[i][j][k]))
        # Normalized velocity model
        if self.is_norm_vms:
            for i in range(len(velocity_models)):
                for j in range(velocity_models[i].shape[0]):
                    velocity_models[i][j][0] = utils.minmax_normalize(velocity_models[i][j][0],
                                                                      np.min(velocity_models[i][j][0]),
                                                                      np.max(velocity_models[i][j][0]))
        # Blur velocity model
        if self.is_blur_vms:
            for i in range(len(velocity_models)):
                for j in range(velocity_models[i].shape[0]):
                    velocity_models[i][j][0] = skimage.filters.gaussian(velocity_models[i][j][0], 5)

    def get_dataset_loader(self):
        """
        Getter
        """
        sampler = RandomSampler(self.fwi_dataset)
        dataset_loader = DataLoader(
            self.fwi_dataset, batch_size=self.batch_size,
            sampler=sampler, pin_memory=True, drop_last=True)
        return dataset_loader

    def network_train_simple_net(self, para_data_zip: list, epoch_id: int, iteration: int, step: int, all_epoch: int):
        """
        One training operation for InversionNet, DDNet70 and LInvNet

        :param para_data_zip:   Packaged list of data files
        :param epoch_id:        Which epoch is the current training belonging to
        :param iteration:       Iteration record number for the current training
        :param step:            Number of batches to feed to train the entire dataset
        :param all_epoch:       Total number of epochs in the current training
        :return:                Training loss object (If it is DDNet70, output the main loss in the joint loss function)
        """

        data_zip = para_data_zip

        self.net_model.train()

        if torch.cuda.is_available():
            for p in range(len(data_zip)):
                data_zip[p] = data_zip[p].cuda(non_blocking=True)

        self.optimizer.zero_grad()

        if self.network_name in ["InversionNet", "LInvNet", "SeisDeepNET70"]:
            pre_velocity_models = self.net_model(data_zip[0])
            loss = self.criterion(pre_velocity_models, data_zip[1])

        elif self.network_name == "DDNet70":
            pre_velocity_models = self.net_model(data_zip[0])
            loss = self.criterion(pre_velocity_models[0], pre_velocity_models[1], data_zip[1], data_zip[2])

        elif self.network_name == "LInvNet":
            pre_velocity_models = self.net_model(data_zip[0])
            loss = self.criterion(pre_velocity_models, data_zip[1])

        else:
            print('The "model_keyword" parameter selected in the one_epoch_train(...) '
                  'is the undefined network model keyword! Please check!')
            exit(0)

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        loss.backward()
        self.optimizer.step()

        if iteration % self.display_step == 0:
            print('[{}][{}] Epochs: {}/{}, Iteration: {}/{} --- Training Loss: {:.6f} -- LR: {:.6f}'
                  .format(self.network_name, self.fwi_dataset.dataset_name,
                          epoch_id, all_epoch,
                          iteration, step * all_epoch,
                          loss.item(), self.optimizer.param_groups[0]['lr']))

        if self.network_name in ["DDNet70", "ABAFWI"]:
            return self.criterion.mse
        else:
            return loss

    def network_train_denseinvnet(self, para_data_zip, epoch_id, iteration, step, all_epoch):
        """
        One training operation for DenseInvNet

        :param para_data_zip:   Packaged list of data files
        :param epoch_id:        Which epoch is the current training belonging to
        :param iteration:       Iteration record number for the current training
        :param step:            Number of batches to feed to train the entire dataset
        :param all_epoch:       Total number of epochs in the current training
        :return:                L_main in joint loss function
        """
        data_zip = para_data_zip
        cur_lpips = -1          # Determine if LPIPS has been introduced (-1 indicates no LPIPS constraint)

        if self.args.age_param_list[self.args.cur_stage] != -1 and epoch_id % 2 == 0:
            self.net_model.eval()
            batch_lpips_sum = 0.0

            temp_common_shot_gathers = data_zip[0].cuda(non_blocking=True)
            temp_velocity_models = data_zip[1].cuda(non_blocking=True)
            temp_lowres_models = data_zip[2].cuda(non_blocking=True)
            temp_pre_velocity_models, _ = self.net_model(temp_lowres_models, temp_common_shot_gathers)

            for bt in range(self.batch_size):
                batch_lpips_sum += self.lpips_object.forward(temp_velocity_models[bt, 0, ...].cpu(),
                                                             temp_pre_velocity_models[bt, 0, ...].cpu()).item()
            cur_lpips = batch_lpips_sum / self.batch_size   # Obtain the mean of LPIPS

        self.net_model.train()

        if torch.cuda.is_available():
            for p in range(len(data_zip)):
                data_zip[p] = data_zip[p].cuda(non_blocking=True)

        self.optimizer.zero_grad()

        pre_velocity_models = self.net_model(data_zip[2], data_zip[0])

        loss = self.criterion(pre_velocity_models[0], pre_velocity_models[1], data_zip[1], data_zip[3])

        if cur_lpips != -1:
            if self.args.is_spl:    # Determine if SPL has been introduced in the LPIPS constraint
                sigma = self.args.age_param_list[self.args.cur_stage]   # Obtain age parameters "sigma"
                right_boundary = 0.18
                left_boundary = 0.0
                self.gauss_spl_func = lambda x: math.exp(
                    -(2 * (x - left_boundary) / sigma / (right_boundary - left_boundary)) ** 2)

                spl_lpips = self.gauss_spl_func(cur_lpips)  # Obtain learning weights through the v*(LPIPS) function

                gamma_spl_lpips = self.args.param_list[self.args.cur_stage][2] * spl_lpips  # δ × v*(LPIPS)

                # loss = loss + gamma_spl_lpips * cur_lpips   # + δ × v*(LPIPS) × Llp
                loss = loss + gamma_spl_lpips * self.criterion.mse  # + δ × v*(LPIPS) × Lmain

            else:
                # loss = loss + self.args.param_list[self.args.cur_stage][2] * criterion.mse
                loss = loss + self.args.param_list[self.args.cur_stage][2] * cur_lpips  # + δ × LPIPS

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        loss.backward()
        self.optimizer.step()

        if iteration % self.display_step == 0:
            print('[{}][{}] Epochs: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'
                  .format(self.network_name, self.fwi_dataset.dataset_name,
                          epoch_id, all_epoch,
                          iteration, step * all_epoch,
                          loss.item()), end="\t")
            if cur_lpips != -1:
                if self.criterion.cross == 0:
                    temp_cross_loss = 0.0
                else:
                    temp_cross_loss = self.criterion.cross.item()
                if self.args.is_spl:
                    print("MSE: {} × {:.5f}, CRO: {} × {:.4f}, SPL_LIPIS: {} × SPL({:.3f}) × {:.5f}".format(
                        self.args.param_list[self.args.cur_stage][0], self.criterion.mse.item(),
                        self.args.param_list[self.args.cur_stage][1], temp_cross_loss,
                        self.args.param_list[self.args.cur_stage][2], spl_lpips, self.criterion.mse))
                else:
                    print("MSE: {} × {:.5f}, CRO: {} × {:.4f}, LIPIS: {} × {:.4f}".format(
                        self.args.param_list[self.args.cur_stage][0], self.criterion.mse.item(),
                        self.args.param_list[self.args.cur_stage][1], temp_cross_loss,
                        self.args.param_list[self.args.cur_stage][2], cur_lpips))
            else:
                print("")

        return self.criterion.mse

    def network_train_velocitygan(self, para_data_zip, epoch_id, iteration, step, all_epoch, is_train_g=False):
        """
        One training operation for VelocityGAN

        :param para_data_zip:   Packaged list of data files
        :param epoch_id:        Which epoch is the current training belonging to
        :param iteration:       Iteration record number for the current training
        :param step:            Nu  mber of batches to feed to train the entire dataset
        :param all_epoch:       Total number of epochs in the current training
        :param is_train_g:      Whether to train the generator
        :return:                The l2 loss of the generator
                                (If the generator is not trained in this epoch then output None)
        """
        data_zip = para_data_zip

        self.net_model.train()
        self.net_model_D.train()

        if torch.cuda.is_available():
            for p in range(len(data_zip)):
                data_zip[p] = data_zip[p].cuda(non_blocking=True)

        # Fix the generator and update the discriminator first
        self.optimizer_D.zero_grad()

        with torch.no_grad():  # The generator does not take derivatives
            pre_velocity_models = self.net_model(data_zip[0])
        loss_d, loss_diff, loss_gp = self.criterion_D(data_zip[1], pre_velocity_models, self.net_model_D)
        loss_d.backward()
        self.optimizer_D.step()

        # Regularly update the generator
        if is_train_g:
            self.optimizer.zero_grad()
            pred_seismic_data = self.net_model(data_zip[0])
            loss_g, loss_g1v, loss_g2v = self.criterion(pred_seismic_data, data_zip[1], self.net_model_D)
            loss_g.backward()
            self.optimizer.step()

            if np.isnan(float(loss_g.item())):
                raise ValueError('loss is nan while training')

            print('[{}][{}] Epochs: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'
                  .format(self.network_name, self.fwi_dataset.dataset_name,
                          epoch_id, all_epoch,
                          iteration, step * all_epoch,
                          loss_g2v.item()))
            return loss_g2v

        return None

    def network_test(self, para_data_zip):
        """
        Perform an inversion of the data using the network

        :param para_data_zip:   Packaged list of data files
        :returns:               Velocity model obtained from the inversion and target velocity model
        """
        data_zip = para_data_zip
        self.net_model.eval()

        if torch.cuda.is_available():
            for p in range(len(data_zip)):
                if isinstance(data_zip[p], torch.Tensor):
                    data_zip[p] = data_zip[p].cuda(non_blocking=True)
                elif isinstance(data_zip[p], np.ndarray):
                    data_zip[p] = torch.from_numpy(data_zip[p][np.newaxis, ...]).cuda(non_blocking=True)
                else:
                    print("unknown parameters!")
                    exit(0)
        if self.network_name in ["InversionNet", "VelocityGAN", "LInvNet", "SeisDeepNET70"]:
            pre_velocity_models = self.net_model(data_zip[0])
        elif self.network_name == "DDNet70":
            pre_velocity_models, _ = self.net_model(data_zip[0])
        elif self.network_name == "DenseInvNet":
            pre_velocity_models, _ = self.net_model(data_zip[2], data_zip[0])

        else:
            print('The "model_keyword" parameter selected in the one_epoch_train(...) '
                  'is the undefined network model keyword! Please check!')
            exit(0)

        pre_velocity_models = pre_velocity_models.cpu().detach().numpy()
        pre_velocity_models = np.where(pre_velocity_models > 0.0, pre_velocity_models, 0.0)  # Delete bad points
        gt_velocity_models = data_zip[1].cpu().detach().numpy()

        return pre_velocity_models, gt_velocity_models


def example_test():
    """
    An Example of using the NetworkControl class

    :return:
    """

    args = parse_args("-n CurveVelA -net DDNet70 -rd ./configuration/cva_cont.txt")
    temp_dataset = FWIDataset(args)
    temp_dataset.load2memory()
    temp_net = NetworkControl(temp_dataset, args)


if __name__ == '__main__':
    example_test()

