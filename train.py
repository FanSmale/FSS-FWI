import time
import numpy as np
import warnings

from parse_args import *
from fwi_dataset import FWIDataset
from net_control import NetworkControl


class TrainingLogger(object):
    """
    A series of loss record
    """

    def __init__(self):
        self.train_loss_records = []
        self.time_records = []
        self.valid_loss_records = []
        self.vtime_records = []

    def update(self, para_loss: float, para_time: float, para_vloss: float = None, para_vtime: float = None):
        """
        Include loss value in statistical list

        :param para_loss:   Current training set loss
        :param para_time:   Current training runtime
        :param para_vloss:  Current validation set loss
        :param para_vtime:  Current validation runtime
        :return:
        """
        self.train_loss_records.append(para_loss)
        self.time_records.append(para_time)
        if para_vloss is not None:
            self.valid_loss_records.append(para_vloss)
            self.vtime_records.append(para_vtime)

    def print_new(self):
        """
        Display current loss and running time information

        :return:
        """
        print("----- Current Epoch Finished ! -----")
        print("Loss: {:.6f}".format(self.train_loss_records[-1]))
        print("Consuming time: {:.0f} m {:.0f} s".format(self.time_records[-1] // 60, self.time_records[-1] % 60))
        if self.valid_loss_records is not []:
            print("Valid loss: {:.6f}".format(self.valid_loss_records[-1]))
            print("Valid consuming time: {:.0f} s".format(self.vtime_records[-1]))
        print("------------------------------------")

    def save_loss(self, save_path: str):
        """
        Save the current training set loss record

        :param save_path:   Training Loss curve saving path
        :return:
        """
        np.save(save_path, np.array(self.train_loss_records))
        print("Loss records saved.")

    def save_vloss(self, save_path: str):
        """
        Save the current validation set loss record

        :param save_path:   Validation loss curve saving path
        :return:
        """
        if self.valid_loss_records is not []:
            np.save(save_path, np.array(self.valid_loss_records))
            print("Validation Loss records saved.")


class Trainer(object):
    """
    A series of network training
    """

    def __init__(self, para_args: argparse):
        """

        :param para_args:   Object to store global variables
        """

        # Get global parameters
        self.args = para_args
        self.training_keywords = para_args.training_keywords
        self.epoch = para_args.epoch
        self.save_time = para_args.save_time
        self.datasets = FWIDataset(self.args)
        self.datasets.load2memory()
        self.net_control = NetworkControl(self.datasets, self.args)

        # Initialize new values
        if self.epoch % self.save_time != 0:
            warnings.warn("Warning: During the training process, "
                          "the model saved does not cover the last epoch.", UserWarning)
        self.save_epoch = self.epoch // self.save_time
        self.step = int(len(self.datasets) // self.net_control.batch_size)
        self.logger = TrainingLogger()

        # Set validation set
        if self.args.is_validation:
            print("Validation set is building...")
            vargs = self.args
            vargs.is_training = False
            vargs.test_size = 500
            vargs.read_range = [0]
            self.validation_set = FWIDataset(vargs)
            self.validation_set.load2memory()
            self.validation_control = NetworkControl(self.validation_set, vargs)
            self.validation_control.net_model = self.net_control.net_model
            self.validation_control.optimizer = self.net_control.optimizer

    def one_epoch_train(self, epoch_id: int):
        """
        Operations during a epoch of training

        :param epoch_id:    Which epoch of training is the current network in
        :returns:           Average value of losses and runtime
        """

        temp_loss_list = []
        temp_time = time.time()

        for index, data_zip in enumerate(self.net_control.dataset_loader):
            iteration = epoch_id * self.step + index + 1
            if args.network_name == "DenseInvNet":
                loss = self.net_control.network_train_denseinvnet(data_zip, epoch_id, iteration, self.step, self.epoch)
                temp_loss_list.append(loss.item())

            elif args.network_name == "VelocityGAN":
                if ((index + 1) % self.args.n_critic == 0) or (index == len(self.net_control.dataset_loader) - 1):
                    loss = self.net_control.network_train_velocitygan(data_zip, epoch_id, iteration, self.step,
                                                                      self.epoch, is_train_g=True)
                    temp_loss_list.append(loss.item())
                else:
                    self.net_control.network_train_velocitygan(data_zip, epoch_id, iteration, self.step, self.epoch)
            else:
                loss = self.net_control.network_train_simple_net(data_zip, epoch_id, iteration, self.step, self.epoch)
                temp_loss_list.append(loss.item())

        return np.array(temp_loss_list).mean(), time.time() - temp_time

    def multi_epoch_train(self):
        """
        Operations during multiple epochs of training

        :return:
        """

        for temp_epoch in range(self.epoch):

            if temp_epoch < self.args.begin:
                continue

            try:
                self.args.cur_stage = self.args.epoch_checkpoints_list.index(temp_epoch)
            except:
                pass

            one_epoch_avgloss, temp_time = self.one_epoch_train(temp_epoch)
            vloss, vtime = None, None
            if self.args.is_validation:
                vloss, vtime = self.get_validation_info()

            self.logger.update(one_epoch_avgloss, temp_time, vloss, vtime)
            self.logger.print_new()

            self.logger.save_loss("./results/{}Results/{}LossRecords_{}.npy".format(self.datasets.dataset_name,
                                                                                    self.training_keywords,
                                                                                    self.net_control.network_name))
            self.logger.save_vloss("./results/{}Results/{}VLossRecords_{}.npy".format(self.datasets.dataset_name,
                                                                                      self.training_keywords,
                                                                                      self.net_control.network_name))
            if (temp_epoch + 1) % self.save_epoch == 0:
                self.net_control.network_save("./models/{}Model/{}{}_{}of{}.pkl".format(self.datasets.dataset_name,
                                                                                        self.training_keywords,
                                                                                        self.net_control.network_name,
                                                                                        temp_epoch + 1, self.epoch))

    def get_validation_info(self):
        """
        Feed the validation set into the network to run a validation and return loss and runtime information

        :returns:    Average value of losses and runtime
        """

        temp_loss = 0.0
        temp_time = time.time()

        for index, data_zip in enumerate(self.validation_control.dataset_loader):
            pre_vmodels, gt_vmodels = self.validation_control.network_test(data_zip)
            temp_loss += np.mean((pre_vmodels - gt_vmodels) ** 2)

        return temp_loss / (index + 1), time.time() - temp_time


if __name__ == '__main__':

    ###############################################################
    # Some commands commonly used for training different networks #
    ###############################################################

    # Simulation training of 500 data: (Used to quickly check some problems in the program)
    # -n CurveVelA -net InversionNet -ep 10 -T -r 0 1 -trb 50 -tr 500

    # Training for InversionNet
    # -n CurveVelA -net InversionNet -T -st 12 -V

    # Training for VelocityGAN
    # -n CurveVelA -net VelocityGAN -T -ep 480 -st 12 -V

    # Training for DDNet70
    # -n CurveVelA -net DDNet70 -T -st 12 -V

    # Training for LResInvNet
    # -n CurveVelA -net LResInvNet -T -ep 100 -st 10 -V -bv

    # Training for DenseInvNet (\alpha = \beta = 0.5, without LPIPS, without DP)
    # -n CurveVelA -net DenseInvNet -T -st 12 -V -mc
    # -apl -1 -1 -1 -1
    # -pl 0.5_0.5_0.0 0.5_0.5_0.0 0.5_0.5_0.0 0.5_0.5_0.0

    # Training for DenseInvNet (\alpha = \beta = \delta = 0.5, with LPIPS, without DP)
    # -n CurveVelA -net DenseInvNet -T -st 12 -V -kw [LPIPS] -mc
    # -apl 1 1 1 1
    # -pl 0.5_0.5_0.5 0.5_0.5_0.5 0.5_0.5_0.5 0.5_0.5_0.5

    # Training for DenseInvNet (with LPIPS, with DP)
    # -n CurveVelA -net DenseInvNet -T -st 12 -V -kw [LPIPS+DP] -mc

    # Training for DenseInvNet (with CSPL, with DP)
    # -n CurveVelA -net DenseInvNet -T -st 12 -V -kw [CSPL+DP] -mc -spl

    # Example: Continue training from epoch 70 on the existing network file (.pkl)
    # -st 24 -beg 70 -nep -spl .\models\CurveFaultAModel\[CSPL+DP]DenseInvNet_70of120.pkl

    ###############################################################

    args = parse_args("-n CurveVelA -net DenseInvNet -T -st 12 -V -kw [CSPL+DP] -mc -spl")
    temp_trainer = Trainer(args)
    temp_trainer.multi_epoch_train()
