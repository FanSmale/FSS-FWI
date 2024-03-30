from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity

import torch
import lpips
import numpy as np


class Metrics(object):
    """
    A series of processing of metrics
    """

    def __init__(self):

        self.prediction = None
        self.ground_true = None
        self.lpips_object = lpips.LPIPS(net='alex', version="0.1")
        self.metrics_dict = {
            "MSE": [],
            "MAE": [],
            "LPIPS": [],
            "PSNR": [],
            "UIQ": [],
            "SSIM": []

        }

    def print_metrics_multiline(self):
        """
        Multi-line printing
        """
        for metric in self.metrics_dict:
            print("{}: {:.6f}".format(metric, self.metrics_dict[metric][-1]))

    def print_metrics_singleline(self, k):
        """
        Single-line printing

        :param k:    Number for display
        :return:
        """
        print("The {} testing: ".format(k), end="")
        for metric in self.metrics_dict:
            print("\t{}: {:.6f}".format(metric, self.metrics_dict[metric][-1]), end="")
        print("")

    def get_metrics(self):
        """
        Getter
        """
        return self.metrics_dict

    def update_samples(self, para_prediction: np.ndarray, para_ground_true: np.ndarray):
        """
        Update each of the current metrics to the metrics list

        :param para_prediction:     Velocity model obtained from the inversion
        :param para_ground_true:    Target velocity model
        :return:
        """
        self.prediction = para_prediction
        self.ground_true = para_ground_true
        self.metrics_dict["MSE"].append(self.run_mse())
        self.metrics_dict["MAE"].append(self.run_mae())
        self.metrics_dict["UIQ"].append(self.run_uiq())
        self.metrics_dict["LPIPS"].append(self.run_lpips())
        self.metrics_dict["SSIM"].append(self.run_ssim())
        self.metrics_dict["PSNR"].append(self.run_psnr())

    def avg_dict(self):
        """
        Average the values in each list in the dictionary
        """
        for metric in self.metrics_dict:
            self.metrics_dict[metric] = np.mean(np.array(self.metrics_dict[metric]))

    def run_mse(self):
        """
        MSE metric
        """
        return np.mean((self.prediction - self.ground_true) ** 2)

    def run_mae(self):
        """
        MAE metric
        """
        return np.mean(np.abs(self.prediction - self.ground_true))

    def run_uiq(self, ws=8):
        """
        UIQ metric

        :param ws:  Window size
        :return
        """
        N = ws ** 2

        GT_sq = self.ground_true * self.ground_true
        P_sq = self.prediction * self.prediction
        GT_P = self.ground_true * self.prediction

        GT_sum = uniform_filter(self.ground_true, ws)
        P_sum = uniform_filter(self.prediction, ws)
        GT_sq_sum = uniform_filter(GT_sq, ws)
        P_sq_sum = uniform_filter(P_sq, ws)
        GT_P_sum = uniform_filter(GT_P, ws)

        GT_P_sum_mul = GT_sum * P_sum
        GT_P_sum_sq_sum_mul = GT_sum * GT_sum + P_sum * P_sum
        numerator = 4 * (N * GT_P_sum - GT_P_sum_mul) * GT_P_sum_mul
        denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
        denominator = denominator1 * GT_P_sum_sq_sum_mul

        q_map = np.ones(denominator.shape)
        index = np.logical_and((denominator1 == 0), (GT_P_sum_sq_sum_mul != 0))
        q_map[index] = 2 * GT_P_sum_mul[index] / GT_P_sum_sq_sum_mul[index]
        index = (denominator != 0)
        q_map[index] = numerator[index] / denominator[index]

        s = int(np.round(ws/2))
        return np.mean(q_map[s: -s, s: -s])

    def run_lpips(self):
        """
        LPIPS metric
        """
        GT_tensor = torch.from_numpy(self.ground_true).float()
        P_tensor = torch.from_numpy(self.prediction).float()
        return self.lpips_object.forward(GT_tensor, P_tensor).item()

    def run_ssim(self):
        """
        SSIM metric
        """
        return structural_similarity(self.ground_true, self.prediction, win_size=11)

    def run_psnr(self):
        """
        PSNR metric
        """
        mse = np.mean((self.prediction - self.ground_true) ** 2)
        return 10 * np.log10(1 / mse)


if __name__ == '__main__':
    pass
