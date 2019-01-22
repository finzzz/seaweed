from PIL import Image as im
from PIL import ImageEnhance as en
import numpy as np


def random_multiplier(amount):
    return np.random.choice(amount)


def hue_shift(img, amount):
    # img = PIL Image object
    hsv_img = img.convert('HSV')
    hsv = np.asarray(hsv_img)
    hsv.setflags(write=1)
    hsv[..., 0] = (hsv[..., 0]*amount) % 360
    new_img = im.fromarray(hsv, 'HSV')
    return new_img.convert('RGBA')


def color_shift(img, amount):
    # img = PIL Image object
    enh = en.Color(img)
    return enh.enhance(amount)


def downscale(img, factor=8):
    # img = PIL Image object
    width = int(img.width/factor)
    height = int(img.height/factor)
    return img.resize([width, height], resample=im.BICUBIC)


def preprocess(img_dir, dims, channels, mult, m_true=1):
    y = im.open(img_dir)

    if channels == 3:
        y = y.convert('RGB')
    elif channels == 4:
        y = y.convert('RGBA')

    y = y.resize((dims.get("width"), dims.get("height")), resample=im.BICUBIC)

    factor = random_multiplier(mult)
    x = color_shift(y, factor)
    y = color_shift(y, m_true)

    x = np.asarray(x, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)

    return x, y
