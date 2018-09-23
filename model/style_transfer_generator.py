import logging
import time
import ntpath
from argparse import Namespace
import scipy.misc

import numpy as np
from scipy.optimize import fmin_l_bfgs_b as optimizer

from model.loss_evaluator import LossEvaluator


class StyleTransferGenerator(object):
    def __init__(self, input_file: str, input_image: np.ndarray, style_file: str, style_image: np.ndarray,
                 args: Namespace) -> None:
        self.__logger = logging.getLogger(__name__)
        self.__args = args
        self.__input_img = input_image
        self.__input_file = input_file
        self.__style_img = style_image
        self.__style_file = style_file
        self.__loss_evaluator = LossEvaluator(self.__input_img, self.__style_img, args)

    def generate(self):
        x = np.copy(self.__input_img).flatten()

        for i in range(self.__args.epochs):
            self.__logger.info('Epoch: {} started'.format(i))
            start_time = time.time()
            x, min_val, info = optimizer(self.__loss_evaluator.loss, x,
                                         fprime=self.__loss_evaluator.gradients,
                                         maxfun=self.__args.max_fun,
                                         disp=self.__args.verbose)
            self.__logger.info('Loss for epoch {} has value: {}'.format(i, min_val))

            # Save current generated image
            img = x.copy().reshape((self.__input_img.shape[1], self.__input_img.shape[2], 3))
            img = np.clip(img, 0, 255).astype(np.uint8)

            input_filename = self.__get_file_name(self.__input_file)
            style_filename = self.__get_file_name(self.__style_file)
            output_filename = "{}/{}_{}_iteration_{}.png".format(self.__args.output_dir,
                                                                 input_filename,
                                                                 style_filename,
                                                                 i)

            scipy.misc.imsave(output_filename, img)
            self.__logger.info("For epoch {} current image is saved in: ".format(i, output_filename))
            end_time = time.time()
            self.__logger.info('Iteration {} completed in {}secs'.format(i, round(end_time - start_time, 4)))

    @classmethod
    def __get_file_name(cls, path) -> str:
        head, tail = ntpath.split(path)
        value: str = tail or ntpath.basename(head)
        return value.split('.')[0]
