import os
import argparse
import json


def parse_args(command_string: str):

    """
    Parse command strings into meaningful global parameters

    :param command_string:  Command string to control running
    :return:                Argparse object
    """

    parser = argparse.ArgumentParser(description='DL-FWI Experiment')
    command_list = command_string.split()

    if "-net" in command_list:
        pos = command_list.index("-net")
        net_name = command_list[pos + 1]
    else:
        net_name = "InversionNet"

    data_config = './configuration/network_config.json'
    assert os.path.exists(data_config), 'Config file is not accessible.'
    with open(data_config) as f:
        net_cfg = json.load(f)[net_name]

    if "-n" in command_list:
        pos = command_list.index("-n")
        dataset_name = command_list[pos + 1]
    else:
        dataset_name = "CurveVelA"

    data_config = './configuration/dataset_config.json'
    assert os.path.exists(data_config), 'Config file is not accessible.'
    with open(data_config) as f:
        dset_cfg = json.load(f)[dataset_name]

    # for all
    parser.add_argument('-V', '--is_validation', action='store_true', help='Whether to introduce validation set.')

    # for net_control.py
    parser.add_argument('-net', '--network_name', default=net_name, type=str, help='The name of network.')
    parser.add_argument('-nep', '--network_path', default="", type=str, help='Path to network files.')
    parser.add_argument('-lr', '--learning_rate', default=net_cfg["common"]["learning_rate"], type=float, help='The learning rate of network.')
    parser.add_argument('-trb', '--train_batch_size', default=net_cfg["common"]["train_batch_size"], type=int, help='The amount of data fed into the network during one training.')
    parser.add_argument('-teb', '--test_batch_size', default=net_cfg["common"]["test_batch_size"], type=int, help='The amount of data fed into the network during one testing.')
    parser.add_argument('-c', '--in_channel', default=net_cfg["common"]["in_channel"], type=int, help='The channel size of the network input.')
    parser.add_argument('-nc', '--is_norm_csg', action='store_false', help='No normalizing the seismic data.')
    parser.add_argument('-nv', '--is_norm_vms', action='store_false', help='No normalizing the velocity models.')
    parser.add_argument('-bv', '--is_blur_vms', action='store_true', help='Apply Gaussian blur to the velocity models.')
    parser.add_argument('-mc', '--is_mute_csg', action='store_true', help='Muting direct waves from seismic data.')
    parser.add_argument('-nocuda', '--is_use_cuda', action='store_false', help='Do not use cuda.')

    # for fwi_dataset.py
    parser.add_argument('-n', '--dataset_name', default=dataset_name, type=str, help='The name of dataset.')
    parser.add_argument('-shot', '--shot_num', default=dset_cfg["shot_num"], type=int, help='How many shots have been sampled.')
    parser.add_argument('-sshape', '--seismic_shape', nargs='+', type=int, default=dset_cfg['seismic_shape'], help='Dimensions of a seismic data file.')
    parser.add_argument('-vshape', '--vmodel_shape', nargs='+', type=int, default=dset_cfg['vmodel_shape'], help='Dimensions of a velocity model file.')
    parser.add_argument('-tr', '--train_size', default=dset_cfg['train_size'], type=int, help='Number of data pair used for training.')
    parser.add_argument('-te', '--test_size', default=dset_cfg['test_size'], type=int, help='Number of data pair used for testing.')
    parser.add_argument('-v', '--data_volume', default=dset_cfg['data_volume'], type=int, help='How much data is stored in a data file.')
    parser.add_argument('-rd', '--readtxt_dir', default=dset_cfg['readtxt_dir'], type=str, help='The location of the txt file that stores the file path.')
    parser.add_argument('-T', '--is_training', action='store_true', help='Determine whether the current data set is used for training or testing.')
    parser.add_argument('-r', '--read_range', nargs='+', type=int, default=0, help='Determine how much data to read into memory.')

    # for train.py
    parser.add_argument('-ep', '--epoch', default=120, type=int, help="Training epoch.")
    parser.add_argument('-st', '--save_time', default=2, type=int, help="How many times do you need to save the network during training.")
    parser.add_argument('-dp', '--display_step', default=2, type=int, help='How many iterations to print once.')
    parser.add_argument('-kw', '--training_keywords', default="", type=str, help="The keywords attached when saving network parameters. It can prevent duplicate name coverage caused by different parameter training.")
    parser.add_argument('-beg', '--begin', type=int, default=0, help="Determine the starting epoch of training")

    # specific parameters of some networks
    abbr = {"CurveVelA": "cva", "CurveFaultA": "cfa", "FlatFaultA": "ffa"}
    if net_name == "DDNet70":
        command_list += ["-rd", "./configuration/{}_cont.txt".format(abbr[dataset_name])]
        parser.add_argument('-lw', '--loss_weight', nargs='+', type=float, default=net_cfg["loss_weight"][dataset_name], help="Training ratio between main task and sub task.")
    elif net_name == "DenseInvNet":
        command_list += ["-rd", "./configuration/{}_lres.txt".format(abbr[dataset_name])]
        parser.add_argument('-ecl', '--epoch_checkpoints_list', nargs='+', type=int, default=net_cfg["epoch_checkpoints_list"][dataset_name], help="Checkpoints at different training stages in dynamic parameter strategy.")
        parser.add_argument('-pl', '--param_list', nargs='+', type=str, default=net_cfg["param_list"][dataset_name], help="The values of the three hyperparameters in the joint loss at different stages.")
        parser.add_argument('-apl', '--age_param_list', nargs='+', type=int, default=net_cfg["age_param_list"][dataset_name], help="Age parameters at different stages.")
        parser.add_argument('-cs', '--cur_stage', type=int, default=-1, help="Determine which training stage the network is currently in.")
        parser.add_argument('-spl', '--is_spl', action='store_true', help="Determine whether to use self paced learning.")
    elif net_name == "VelocityGAN":
        parser.add_argument('-g1v', '--lambda_g1v', type=float, default=net_cfg["loss_param"]["lambda_g1v"])
        parser.add_argument('-g2v', '--lambda_g2v', type=float, default=net_cfg["loss_param"]["lambda_g2v"])
        parser.add_argument('-adv', '--lambda_adv', type=float, default=net_cfg["loss_param"]["lambda_adv"])
        parser.add_argument('-gp', '--lambda_gp', type=float, default=net_cfg["loss_param"]["lambda_gp"])
        parser.add_argument('-lrD', '--lr_discriminator', type=float, default=net_cfg["loss_param"]["lr_discriminator"], help="The learning rate of discriminator.")
        parser.add_argument('-nD', '--n_critic', type=int, default=net_cfg["loss_param"]["n_critic"], help="How many epochs of discriminator need to be trained before training the generator.")

    return parser.parse_args(command_list)
