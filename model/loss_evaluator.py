import logging
from argparse import Namespace

import numpy as np
from keras import backend
from typing import List
from keras.applications import vgg19


class LossEvaluator(object):
    def __init__(self, input_image: np.ndarray, style_image: np.ndarray, args: Namespace) -> None:
        self.__logger = logging.getLogger(__name__)
        self.__style_layers: List[str] = ["block{}_conv1".format(i) for i in range(1, 6)]
        self.__args = args
        self.__input_img = input_image
        self.__style_img = style_image
        self.__loss_value = None
        self.__gradient_values = None
        self.__channels = 3
        self.__batch_size = 1

        self.__img_placeholder = backend.placeholder((self.__batch_size,
                                                      self.__input_img.shape[1],
                                                      self.__input_img.shape[2],
                                                      self.__channels))
        self.__img_preprocessed = vgg19.preprocess_input(self.__img_placeholder)
        self.__model = vgg19.VGG19(input_tensor=self.__img_preprocessed, weights='imagenet', include_top=False)
        self.__model_outputs = backend.function([self.__img_placeholder], [self.__model.outputs[0]])
        if self.__args.verbose:
            self.__model.summary()

        # Function to fetch the values of the current loss and the current gradients
        loss = self.__total_variation()
        self.__fetch_loss_and_grads = backend.function(
            [self.__img_placeholder], [loss, backend.gradients(loss, self.__img_placeholder)[0]])

    def loss(self, x):
        assert self.__loss_value is None
        x = x.reshape((self.__batch_size, self.__input_img.shape[1], self.__input_img.shape[2], self.__channels))
        outs = self.__fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype(np.float64)
        self.__loss_value = loss_value
        self.__gradient_values = grad_values
        return self.__loss_value

    # noinspection PyUnusedLocal
    def gradients(self, x):
        assert self.__loss_value is not None
        copy_gradient = np.copy(self.__gradient_values)
        self.__loss_value = None
        self.__gradient_values = None
        return copy_gradient

    def __total_variation(self):
        init_loss = backend.variable(0.0)
        init_loss += self.__args.variation_weight * self.__loss_total(self.__img_preprocessed)
        init_loss += self.__args.input_weight * self.__loss_input_img()
        init_loss += self.__args.style_weight / len(self.__style_layers) * self.__loss_style_img()

        # Lets calculate the total loss for content image
        calc_loss = backend.function([self.__img_placeholder], [init_loss])
        self.__logger.info("TOTAL LOSS for input image: {}, loss for style image: {}"
                           .format(calc_loss([self.__input_img]), calc_loss([self.__style_img])))
        return init_loss

    def __loss_total(self, x):
        shape = self.__input_img.shape
        a = backend.square(x[:, :shape[1] - 1, :shape[2] - 1, :] - x[:, 1:, :shape[2] - 1, :])
        b = backend.square(x[:, :shape[1] - 1, :shape[2] - 1, :] - x[:, :shape[1] - 1, 1:, :])
        return backend.sum(backend.pow(a + b, 1.25))

    def __loss_input_img(self):
        layer_name = 'block5_conv4'
        get_features = backend.function([self.__img_placeholder],
                                        [self.__model.get_layer(layer_name).output[0]])

        input_img_features = get_features([self.__input_img])[0]
        output_img_features = self.__model.get_layer(layer_name).output[0]
        input_loss = backend.sum(backend.square(output_img_features - input_img_features))

        # If the generated image is the same as content, then input_loss should equal to 0.0
        calc_input_loss = backend.function([self.__img_placeholder], [input_loss])
        self.__logger.info("Input loss for input image: {}, loss for style image: {}"
                           .format(calc_input_loss([self.__input_img]), calc_input_loss([self.__style_img])))
        return input_loss

    def __loss_style_img(self):
        style_img_features = dict()

        for layer_name in self.__style_layers:
            style_img_features[layer_name] = backend.function(
                [self.__img_placeholder],
                [self.__model.get_layer(layer_name).output[0]])([self.__style_img])[0]

        style_loss = backend.variable(0.0)
        for layer_name in self.__style_layers:
            output_img_features = self.__model.get_layer(layer_name).output[0]
            style_loss += self.__compute_style_loss(style_img_features[layer_name], output_img_features)

        calc_style_loss = backend.function([self.__img_placeholder], [style_loss])
        self.__logger.info("Style loss for input image: {}, loss for style image: {}"
                           .format(calc_style_loss([self.__input_img]), calc_style_loss([self.__style_img])))
        return style_loss

    @classmethod
    def __multiply_tensors(cls, x):
        features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
        return backend.dot(features, backend.transpose(features))

    def __compute_style_loss(self, style_image_features, output_image_features):
        s = self.__multiply_tensors(style_image_features)
        o = self.__multiply_tensors(output_image_features)
        size = self.__input_img.shape[2] * self.__input_img.shape[1]
        return backend.sum(backend.square(s - o)) / (4. * (self.__channels ** 2) * (size ** 2))
