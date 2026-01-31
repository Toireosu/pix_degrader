<div>
    <h2 align="center">PIX_DEGRADER</h2>
    <p align="center" style="font-style: italic;">A one file one dependency python script for lowering the quality of images to achieve old school aesthetics.</p>
    <p align="center">made by <a href="https://github.com/Toireosu">William Lindgren ~ Duplo</a></p>  
    <div align="center" style="display: flex; justify-content: center; gap: 40px;">
        <img style="max-width: 300;" src="https://raw.githubusercontent.com/Toireosu/pix_degrader/refs/heads/main/examples/lofi_dude.png"></img>
        <img style="max-width: 300;" src="https://raw.githubusercontent.com/Toireosu/pix_degrader/refs/heads/main/examples/lofi_dude_quake256_linear.png"></img>
    </div>
</div>


## Quickstart

1. Clone this repo: ```git clone https://github.com/Toireosu/pix_degrader.git```\
or simply download the file ```pix_degrader.py```.

2. Make sure you have Pillow installed on your system: ```pip install pillow```.

3. Run ```python pix_degrader.py <some_file>```.

4. Voilà! Your output image will have been rendered to the ```render``` directory next to the script!

## Usage
```
usage: Lowers the quality of images in a retro gaming way :) Outputs in PNG format.

positional arguments:
  INPUT_PATH            the input file or dir to be degraded

options:
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH
                        the output file or dir
  --output_size OUTPUT_SIZE
                        the dimensions of the output image
  --no_dither           disables dithering for values in between colors
  --colorspace_mode {none,bit_depth,image}
                        sets which mode to use when converting colorspace
  --num_bits NUM_BITS   used with colorspace_mode 'bit_depth'
  --color_palette COLOR_PALETTE
                        used with colorspace_mode 'image'
  --skip_non_image_files
                        continues past non-image files without raising an error
  --interpolation_mode {nearest,bilinear}
                        sets which method will be used to interpolate output colors
  --keep_aspect_ratio   forces dimensions of output to the have same aspect ratio as input image, ignores output_size y-value
  -V, --version         show program's version number and exit
``` 


## Dependencies

[Pillow 12.1.0](https://pypi.org/project/pillow/)

## License
MIT-CMU License. See ```License```.