import glob
from PIL import Image
import imageio as iio
import contextlib

img_path = 'results/plots/new_PO2/tidesON/erf_high/*.png'
gif_path = 'results/plots/new_PO2/tidesON/erf_high/contourgif.gif'

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:
    imgs = (stack.enter_context(Image.open(f))
            for f in sorted(glob.glob(img_path)))

    img = next(imgs)

    img.save(fp=gif_path, format='GIF', append_images=imgs,
             save_all=True, duration=250, loop=0)