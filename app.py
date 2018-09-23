import logging
import os

from utils.args_utils import read_args
from utils.image_utils import generate_image, Downloader
from utils.logger_utils import setup_logging


def main():
    setup_logging()
    args = read_args()
    logger = logging.getLogger("main")

    if args.mode == "dir":
        assert os.path.isdir(args.input) and os.path.isdir(args.style)
        input_images = ["{}/{}".format(args.input, f) for f in os.listdir(args.input)]
        style_images = ["{}/{}".format(args.style, f) for f in os.listdir(args.style)]

        logger.info("Input images to proceed: {}, style images to proceed: {}".format(len(input_images),
                                                                                      len(style_images)))
        for i in input_images:
            for s in style_images:
                generate_image(i, s, args)

    elif args.mode == "file":
        assert os.path.isfile(args.input) and os.path.isfile(args.style)
        generate_image(args.input, args.style, args)

    elif args.mode == "url":
        assert args.input.startswith('http://') or args.input.startswith('https://')
        assert args.style.startswith('http://') or args.style.startswith('https://')

        # ext does not matter...
        input_file = "images/input/image.jpg"
        style_file = "images/styles/style.jpg"

        downloader = Downloader()
        downloader.download(args.input, input_file)
        downloader.download(args.style, style_file)

        generate_image(input_file, style_file, args)

    else:
        raise ValueError("Mode can be one of: ['dir', 'file', 'url'] !!!")


if __name__ == "__main__":
    main()
