import argparse
import logging

__logger = logging.getLogger(__name__)


def read_args() -> argparse.ArgumentParser:
    """
    function defines input arguments
    :return: input arguments
    """
    parser = argparse.ArgumentParser(description="AI style transfer allows to transform "
                                                 "input image(s) into output image(s) by using style image(s) "
                                                 "as a pattern.",
                                     epilog="The End!")

    parser.add_argument('-m', '--mode', type=str,
                        choices=["dir", "file", "url"], required=True,
                        help='If "dir" mode, program uses input dir and style dir. '
                             'If "file" mode, you can specify files instead of directories'
                             'If "url" mode, you can specify urls for input and style images')
    parser.add_argument('-i', '--input', default="images/input", type=str,
                        help='For "mode" eq "dir" it is an input images dir. '
                             'For "mode" eq "file" it is a path to input image. '
                             'For "mode" eq "url" it is a url to input image.')
    parser.add_argument('-e', '--epochs', default=8, type=int,
                        help='Number of epochs')
    parser.add_argument('-s', '--style', default="images/styles", type=str,
                        help='For "mode" eq "dir" it is style images dir. '
                             'For "mode" eq "file" it is a path to style image'
                             'For "mode" eq "url" it is a url to style image.')
    parser.add_argument('-o', '--output-dir', default="images/output", type=str,
                        help='Output dir. Default value: "images/output"')
    parser.add_argument('-w', '--width', default=800, type=int,
                        help='Output image width. Height will be calculated based on this value')
    parser.add_argument('-f', '--max-fun', default=30, type=int,
                        help='Maximum number of function evaluations for optimizer')
    parser.add_argument('--input-weight', default=0.1, type=float,
                        help='Input image weight')
    parser.add_argument('--style-weight', default=0.9, type=float,
                        help='Style image weight')
    parser.add_argument('--variation-weight', default=1e-4, type=float,
                        help='Variation weight')
    parser.add_argument('-v', '--verbose', action='count',
                        help='True if verbose, false otherwise. Default value: False')

    if parser.parse_args().verbose:
        __logger.info("Input arguments: {}".format(parser.parse_args()))
    return parser
