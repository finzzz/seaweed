from keras.applications import VGG19
from keras.layers import Conv2D, Input, MaxPool2D
from keras.layers import UpSampling2D, Concatenate, LeakyReLU, ReLU
from keras.models import Model


def conv_lr(layer_x, filters, act=LeakyReLU, use_bias=False):
    y = Conv2D(filters, kernel_size=3, padding="same")(layer_x)
    # layer_x = act(layer_x)
    return y


def unet(height, width, channels):
    # Model
    input_layer = Input(shape=[height, width, channels])
    res = [input_layer]

    # D1
    d1 = conv_lr(input_layer, 48)
    d1 = conv_lr(d1, 48)
    d1 = MaxPool2D()(d1)
    res.append(d1)

    # D2
    d2 = conv_lr(d1, 48)
    d2 = MaxPool2D()(d2)
    res.append(d2)

    # D3
    d3 = conv_lr(d2, 48)
    d3 = MaxPool2D()(d3)
    res.append(d3)

    # D4
    d4 = conv_lr(d3, 48)
    d4 = MaxPool2D()(d4)
    res.append(d4)

    # Mid
    mid = conv_lr(d4, 48)
    mid = MaxPool2D()(mid)
    mid = conv_lr(mid, 48)
    # ------------------------------------------------
    # U1
    u1 = UpSampling2D()(mid)
    u1 = Concatenate(axis=3)([res.pop(), u1])
    u1 = conv_lr(u1, 96)
    u1 = conv_lr(u1, 96)

    # U2
    u2 = UpSampling2D()(u1)
    u2 = Concatenate(axis=3)([res.pop(), u2])
    u2 = conv_lr(u2, 96)
    u2 = conv_lr(u2, 96)

    # U3
    u3 = UpSampling2D()(u2)
    u3 = Concatenate(axis=3)([res.pop(), u3])
    u3 = conv_lr(u3, 96)
    u3 = conv_lr(u3, 96)

    # U4
    u4 = UpSampling2D()(u3)
    u4 = Concatenate(axis=3)([res.pop(), u4])
    u4 = conv_lr(u4, 96)
    u4 = conv_lr(u4, 96)

    # U5
    u5 = UpSampling2D()(u4)
    u5 = Concatenate(axis=3)([res.pop(), u5])
    u5 = conv_lr(u5, 64)
    u5 = conv_lr(u5, 32)

    out = conv_lr(u5, channels, act=ReLU, use_bias=True)

    return Model(input_layer, out)


def vgg_net(height, width, channels):
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]

    img = Input(shape=[height, width, channels])
    img_features = vgg(img)

    return Model(img, img_features)
