import argparse
from enum import Enum
from io import BytesIO
import time
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Any, NamedTuple

class ColorSpaceMode(Enum):
    NONE = "none"
    BIT_DEPTH = "bit_depth"
    FROM_IMAGE = "image"

class InterpolationMode(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"

Size = tuple[int, int]

class Color(NamedTuple):
    r: float
    g: float
    b: float
    a: float = 1.0

    @classmethod
    def from_tuple(cls, t: tuple[float, ...]):
        return cls(
            t[0],
            t[1],
            t[2],
            t[3],
        )
    
    @classmethod
    def from_list(cls, t: list[float]):
        return Color.from_tuple(tuple(t))
    
    def __sub__(self, other: "Color") -> "Color":
        return Color(
            self.r - other.r,
            self.g - other.g,
            self.b - other.b,
            self.a - other.a
        )
    
    def add(self, other: "Color") -> "Color":
        return Color(
            self.r + other.r,
            self.g + other.g,
            self.b + other.b,
            self.a + other.a
        )
    
    def mul(self, scalar: float) -> "Color":
        return Color(
            self.r * scalar,
            self.g * scalar,
            self.b * scalar,
            self.a * scalar
        )

    def dot(self, other: "Color") -> float:
        return ( 
            self.r * other.r + 
            self.g * other.g +
            self.b * other.b +
            self.a * other.a
        )


Bitmap = list[list[Color]]

def size(s: str) -> Size:
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Size must be x,y")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pix_degrader",
        usage="Lowers the quality of images in a retro gaming way :) Outputs in PNG format."
    )

    parser.add_argument(
        "input_path",
        metavar="INPUT_PATH",
        help="the input file or dir to be degraded",
        type=Path
    )

    parser.add_argument(
        "--output_path",
        help="the output file or dir",
        default="render",
        type=Path
    )

    parser.add_argument(
        "--output_size",
        help="the dimensions of the output image",
        default="128,128",
        type=size,
    )

    parser.add_argument(
        "--no_dither",
        help="disables dithering for values in between colors",
        action="store_true",
    )

    parser.add_argument(
        "--colorspace_mode",
        help="sets which mode to use when converting colorspace",
        choices=[x.value for x in list(ColorSpaceMode)],
        default=ColorSpaceMode.BIT_DEPTH.value
    )

    parser.add_argument(
        "--num_bits",
        help="used with colorspace_mode 'bit_depth'",
        type=int,
        default=4
    )

    parser.add_argument(
        "--color_palette",
        help="used with colorspace_mode 'image'",
    )

    parser.add_argument(
        "--skip_non_image_files",
        help="continues past non-image files without raising an error",
        action="store_true"
    )

    parser.add_argument(
        "--interpolation_mode",
        help="sets which method will be used to interpolate output colors",
        choices=[x.value for x in list(InterpolationMode)],
        default=InterpolationMode.NEAREST.value
    )

    parser.add_argument(
        "--keep_aspect_ratio",
        help="forces dimensions of output to the have same aspect ratio as input image, ignores output_size y-value",
        action="store_true"
    )

    parser.add_argument(
        "-V", "--version",
        action="version",
        version="pix_degrader 1.0"
    )

    args = parser.parse_args()

    args.interpolation_mode = InterpolationMode(args.interpolation_mode)
    args.colorspace_mode = ColorSpaceMode(args.colorspace_mode)

    return args

RGBA_BYTES: int = 4
RGB_BYTES: int = 3

def load_image(path: Path, args: argparse.Namespace) -> Bitmap:
    try:
        with Image.open(path) as intermediate:
            width = intermediate.size[0]
            height = intermediate.size[1]
            output: Bitmap = [[] for _ in range(height)]
            intermediate = intermediate.convert("RGBA")
            bytes = intermediate.tobytes()
            size = width * height
            if len(bytes) == size * RGBA_BYTES:
                for i in range(0, len(bytes), RGBA_BYTES):
                    color: Color = Color.from_list([b / 255 for b in bytes[i:i+RGBA_BYTES]])
                    y: int = (i // RGBA_BYTES) // width 
                    output[y].append(color)
            else:
                raise ValueError(f"Unsupported image format!")

            return output
    except UnidentifiedImageError:
        if not args.skip_non_image_files:
            raise 
        return []

def load_input_files(path: Path, args: argparse.Namespace) -> list[Bitmap]:

    paths: list[Path] = []

    if path.is_file():
        paths.append(path)
    else:
        for p in path.glob("**/*"):
            if p.is_file():
                paths.append(p)

    if not paths and not args.skip_non_image_files:
        raise ValueError("no files found to convert")
    elif len(paths) == 1:
        return [load_image(paths[0], args)]
    else:
        imgs: list[Bitmap] = [] 
        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(load_image, p, args) for p in paths]

            for future in as_completed(futures):
                imgs.append(future.result())
            
            return [a for a in imgs if a]
    return []

BAYER_2X2 = [
    [0 / 4, 2 / 4],
    [3 / 4, 1 / 4]
]

def dither_channel(v: float, x: int, y: int, levels: int) -> float:
    scaled = v * (levels - 1)
    base = int(scaled)
    frac = scaled - base

    threshold = BAYER_2X2[y % 2][x % 2]

    if frac > threshold:
        base += 1

    base = max(0, min(base, levels - 1))
    return base / (levels - 1)

def blend_ratio(p: Color, c0: Color, c1: Color) -> float:
    v = c1 - c0
    if v.dot(v) < 1e-6:
        return 0.0
    t = (p - c0).dot(v) / v.dot(v)
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return t

def cdist2(c: Color, v: Color):
    r: float = (c.r - v.r)**2
    g: float = (c.g - v.g)**2
    b: float = (c.b - v.b)**2
    return 0.299*r + 0.587*g + 0.114*b

def apply_filter(img: Bitmap, args: argparse.Namespace, palette: list[Color]) -> Bitmap:
    iw: int = len(img[0])
    ih: int = len(img) 
    ow: int = args.output_size[0]
    oh: int = int(args.output_size[1] * ih / iw) if args.keep_aspect_ratio else args.output_size[1] 
    r_img: Bitmap = [[] for _ in range(0, oh)]

    x_step: float = iw / ow
    y_step: float = ih / oh
    
    # Resize
    for y in range(0, oh):
        for x in range(0, ow):
            iy: int = int(y * y_step)
            ix: int = int(x * x_step)
            px: Color
            if args.interpolation_mode == InterpolationMode.BILINEAR:
                sx: float = (iw - 1) / max(1, ow - 1)
                sy: float = (ih - 1) / max(1, oh - 1)
                fx: float = x * sx
                fy: float = y * sy

                ix: int = int(fx)
                iy: int = int(fy)

                tx: float = fx - ix
                ty: float = fy - iy

                c00: Color = img[iy][ix]
                c10: Color = img[iy][min(ix + 1, iw - 1)]
                c01: Color = img[min(iy + 1, ih - 1)][ix]
                c11: Color = img[min(iy + 1, ih - 1)][min(ix + 1, iw - 1)]

                top: Color = c00.mul(1.0 - tx).add(c10.mul(tx)) 
                bottom: Color = c01.mul(1.0 - tx).add(c11.mul(tx))
                px = top.mul(1.0 - ty).add(bottom.mul(ty))

            elif args.interpolation_mode == InterpolationMode.NEAREST: 
                px = img[iy][ix]
            else:
                raise ValueError("unkown interpolation mode")
            r_img[y].append(px) 
    
    # Limit colorspace

    if args.colorspace_mode != ColorSpaceMode.NONE:
        if args.colorspace_mode == ColorSpaceMode.BIT_DEPTH:
            for y in range(len(r_img)):
                for x in range(len(r_img[0])):
                    levels: int = 2**args.num_bits
                    if args.no_dither:
                        r_img[y][x] = Color.from_list(
                            [ 
                                int(channel * (levels - 1)) / (levels - 1)
                                for channel in r_img[y][x] 
                            ]
                        )
                    else:
                        px = r_img[y][x]
                        r_img[y][x] = Color.from_tuple((
                            dither_channel(px.r, x, y, levels),
                            dither_channel(px.g, x, y, levels),
                            dither_channel(px.b, x, y, levels),
                            px[3]
                        ))
        if args.colorspace_mode == ColorSpaceMode.FROM_IMAGE:
            for y in range(len(r_img)):
                for x in range(len(r_img[0])):
                    color: Color | None = None 
                    second_nearest: Color | None = None
                    min_dist: float = 10000
                    px = r_img[y][x]
                    for c in palette:
                        dist: float = cdist2(c, px)
                        if min_dist > dist:
                            second_nearest = color
                            min_dist = dist
                            color = c

                    if color:
                        if args.no_dither:
                            r_img[y][x] = color
                        elif second_nearest:
                            t = blend_ratio(px, color, second_nearest)
                            b = BAYER_2X2[y % 2][x % 2]
                            if t > b:
                                r_img[y][x] = second_nearest
                            else:
                                r_img[y][x] = color
                        else:
                            r_img[y][x] = color
                    else:
                        raise ValueError("could not find matching color in palette")

    return r_img 

def apply_filters(imgs: list[Bitmap], args: argparse.Namespace, palette: list[Color]) -> list[Bitmap]:
    if not imgs: 
        return []
    elif len(imgs) == 1:
        return [apply_filter(imgs[0], args, palette)]
    else:
        f_imgs: list[Bitmap] = [] 
        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(apply_filter, img, args, palette) for img in imgs]

            for future in as_completed(futures):
                f_imgs.append(future.result())
            
            return f_imgs

def convert_back(img: Bitmap) -> bytes:
    pil_img = Image.new("RGBA", (len(img[0]), len(img)))
    for y, xs in enumerate(img): 
        for x, xv in enumerate(xs):
            pil_img.putpixel((x, y), tuple([ int(c * 255) for c in xv ]))

    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()

def print_files(imgs: list[Bitmap], args: argparse.Namespace):
    
    img_datas: list[bytes] = []
    if not imgs:
        return
    elif len(imgs) == 1:
        img_datas.append(convert_back(imgs[0]))
    else:
        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(convert_back, img) for img in imgs]

            for future in as_completed(futures):
                img_datas.append(future.result())

    output_path: Path = args.output_path
    if output_path.is_file():
        base_name = output_path.name.split('.')[0]
        output_path = output_path.parent
    else:
        base_name = "img_"

    output_path.mkdir(parents=True, exist_ok=True)
    base_name: str
    for i, data in enumerate(img_datas):
        full_path: Path = output_path 
        if output_path.is_file() and i == 0:
            full_path = full_path.joinpath(base_name)
        else:
            full_path = full_path.joinpath(f"{base_name}{i}.png")
        
        with full_path.open(mode = "wb") as file:
            file.write(data)

def wait_for_worker(future: Future[Any]):
    secs: int = 0
    time.sleep(0.1) # To catch the case where the thread finishes immediately
    while not future.done():
        time.sleep(1.0)
        dots = "." * (secs % 3 + 1)
        print(f"working{dots} ({secs}s)")
        secs += 1
    print(f"took {secs}s")

def main():
    args = parse_args()

    print("==== PIX_DEGRADER 1.0 ====")
    print(" ======================== ")
    print("\n ")


    with ThreadPoolExecutor() as ex:
        print("loading input files")
        future = ex.submit(load_input_files, args.input_path, args)
        wait_for_worker(future)
        imgs = future.result()    

        print("\n")
        
        print("applying filters")
        palette: list[Color] = []
        if args.color_palette:
            palette = list(set([
                c 
                for row in load_image(args.color_palette, args)
                for c in row 
            ]))

        future = ex.submit(apply_filters, imgs, args, palette)
        wait_for_worker(future)
        imgs = future.result()

        print("\n")
        print("writing output")
        future = ex.submit(print_files, imgs, args)
        wait_for_worker(future)

        print("\n")
        print("Jobs done :3")
        print("made by William Lindgren ~ Duplo ~ https://github.com/Toireosu")

if __name__ == "__main__":
    main()