import logging
from argparse import Namespace

import requests
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

from model.style_transfer_generator import StyleTransferGenerator


class Downloader(object):
    def __init__(self) -> None:
        self.__logger = logging.getLogger(__name__)

    def download(self, url: str, output_file: str) -> None:
        """
        downloads file from specified url
        :param url: file to download
        :param output_file: file name. File will be saved in images dir.
        :return: None
        """
        assert url is not None and output_file is not None
        response = requests.get(url, stream=True)
        with open(output_file, "wb") as handle:
            self.__logger.info("Downloading file: {}".format(url))
            for data in tqdm(response.iter_content()):
                handle.write(data)
            self.__logger.info("File downloaded and saved as: {}".format(output_file))


class ImageLoader(object):
    def __init__(self, img_path: str, args: Namespace) -> None:
        """
        ImageLoader allows to load and resize images
        :param img_path: image to load
        :param args: input arguments
        """
        w, h = load_img(img_path).size
        self.__logger = logging.getLogger(__name__)
        self.__args = args
        self.__img_width = args.width
        self.__img_height = int(h * self.__img_width / w)
        if args.verbose:
            self.__logger.info("File '{}' input size = ({},{}), output size = ({},{})".format(img_path, w, h,
                                                                                              self.__img_width,
                                                                                              self.__img_height))

    def load_and_resize_image(self, image_path):
        img = load_img(image_path, target_size=(self.__img_height, self.__img_width))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img


def generate_image(input_img: str, style_img: str, args: Namespace):
    """
    function generates image based on input image. Function uses style_image as a pattern
    :param args: program arguments
    :param input_img: input image (to transform)
    :param style_img: style image (style to copy)
    :return: None
    """
    logger = logging.getLogger(__name__)
    img_loader = ImageLoader(input_img, args)
    input_image = img_loader.load_and_resize_image(input_img)
    style_image = img_loader.load_and_resize_image(style_img)
    logger.info("Img '{}' is loaded and will be processed through style '{}'".format(input_img, style_img))
    model = StyleTransferGenerator(input_img, input_image, style_img, style_image, args)
    model.generate()
    logger.info("Img '{}' is generated with style '{}'".format(input_img, style_img))
