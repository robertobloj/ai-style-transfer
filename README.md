# AI style transfer

AI style transfer is a simple project which shows 
how to use artificial intelligence to changing image 
style based on some pattern.

## Requirements

Before we start you have to install: 

- [python 3](https://www.python.org/download/releases/3.0/)
- required packages by following command:

```bash
python -m pip install -r requirements.txt
```

For windows you should install [anaconda](https://www.anaconda.com/download/) instead of standard python distribution.


## Quick start

You can run *app.py* in one of the following modes:

- *dir* - mode, where *app.py* uses *images/input/* dir and *images/styles/* dir 
- *file* - mode, where you can specify files instead of directories
- *url* - mode, where you can specify urls for input and style images. Those files *app.py* downloads from the internet.

Let's assume, we want to choose *file* mode. Then we can invoke *app.py*:

```bash
python app.py -m file -i [INPUT_IMG] -s [STYLE_IMG] -o [OUTPUT_DIR]
```

Where:

- *INPUT_IMG* - path to input image
- *STYLE_IMG* - path to style image
- *OUTPUT_DIR* - output directory, *app.py* will save input image converted through style image into output image

For more check help:

```bash
python app.py -h
```

This command shows you more options:

```bash
TODO
```