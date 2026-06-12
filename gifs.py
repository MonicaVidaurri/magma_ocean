import glob
from PIL import Image
import imageio as iio
import contextlib

img_path = 'results/plots/*.png'
gif_path = 'results/plots/proxb_lowXUV/contourgif.gif'

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:

    # lazily load images
    imgs = (stack.enter_context(Image.open(f))
            for f in sorted(glob.glob(img_path)))

    # extract  first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=gif_path, format='GIF', append_images=imgs,
             save_all=True, duration=250, loop=0)

