# AI style transfer

Project shows how to use artificial intelligence to change image 
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

- *dir* - mode, where *app.py* by default uses *images/input/* dir and *images/styles/* dir 
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

## More options

For more check help:

```bash
python app.py -h
```

This command shows you more options:

```bash
usage: app.py [-h] -m {dir,file,url} [-i INPUT] [-e EPOCHS] [-s STYLE]
              [-o OUTPUT_DIR] [-w WIDTH] [-f MAX_FUN]
              [--input-weight INPUT_WEIGHT] [--style-weight STYLE_WEIGHT]
              [--variation-weight VARIATION_WEIGHT] [-v]

AI style transfer allows to transform input image(s) into output image(s) by
using style image(s) as a pattern.

optional arguments:
  -h, --help            show this help message and exit
  -m {dir,file,url}, --mode {dir,file,url}
                        If "dir" mode, program uses input dir and style dir.
                        If "file" mode, you can specify files instead of
                        directoriesIf "url" mode, you can specify urls for
                        input and style images
  -i INPUT, --input INPUT
                        For "mode" eq "dir" it is an input images dir. For
                        "mode" eq "file" it is a path to input image. For
                        "mode" eq "url" it is a url to input image.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs
  -s STYLE, --style STYLE
                        For "mode" eq "dir" it is style images dir. For "mode"
                        eq "file" it is a path to style imageFor "mode" eq
                        "url" it is a url to style image.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output dir. Default value: "images/output"
  -w WIDTH, --width WIDTH
                        Output image width. Height will be calculated based on
                        this value
  -f MAX_FUN, --max-fun MAX_FUN
                        Maximum number of function evaluations for optimizer
  --input-weight INPUT_WEIGHT
                        Input image weight
  --style-weight STYLE_WEIGHT
                        Style image weight
  --variation-weight VARIATION_WEIGHT
                        Variation weight
  -v, --verbose         True if verbose, false otherwise. Default value: False
```


## TODO

Remove warnings:

```bash
WARNING:tensorflow:Variable += will be deprecated. Use variable.assign_add if you want 
assignment to the variable value or 'x = x + y' if you want a new python Tensor object.
```