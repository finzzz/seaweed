from PIL import Image as im
from PIL import ImageEnhance as en
from PIL import ImageOps as ops
import numpy as np
from sklearn.model_selection import train_test_split as tts
import glob, math
from keras.applications import VGG19
from keras.layers import Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, LeakyReLU,ReLU
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras import metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from scipy.misc import imsave
import argparse

# channels
channels = 3

#random multiplier
# m = [0.5,0.7,0.8,1,1.2,1.3,1.5]
m_true = 0.4 # default 1
m = [0.1,0.2,0.3,0.4]

#default loss: 
l1 = "mean_absolute_error"
l2 = "mean_squared_error"

parser = argparse.ArgumentParser()
parser.add_argument("type", help="train/test model", choices=["train","test"])
parser.add_argument("-mod","--model",help="model.h5 file")
parser.add_argument("-i","--image",help="image to test")
parser.add_argument("-mu","--mult", help="multiplier factor",nargs="+", type=float, default=m)
parser.add_argument("-sh", "--shape", help="image shape [width,height]", type=int, nargs=2, default=[128,96])
parser.add_argument("-l","--loss", help="loss function, l1,l2,l1l2", default=l2)
parser.add_argument("-st","--steps", help="steps per epoch", type=int, default=180)
parser.add_argument("-e","--epoch", help="number of epoch", type=int, default=3)
args = parser.parse_args()

if args.loss == "l1":
    args.loss = l1

if args.type == "test" and not args.model and not args.image:
    parser.error("test needs image & model")

width=args.shape[0]
height=args.shape[1]

def random_multiplier():
    return np.random.choice(args.mult)

def hue_shift(img, amount):
    hsv_img = img.convert('HSV')
    hsv = np.asarray(hsv_img)
    hsv.setflags(write=1)
    hsv[..., 0] = (hsv[..., 0]*amount) % 360
    new_img = im.fromarray(hsv, 'HSV')
    return new_img.convert('RGBA')
# a = im.open('5.png')
# a = hue_shift(a,2)


def color_shift(img, amount):
    enh = en.Color(img)
    return enh.enhance(amount)
# a = im.open('60m.jpg')
# a = color_shift(a,0.6)
# a.show()


def downscale(img, factor=8):
    width = int(img.width/factor)
    height = int(img.height/factor)
    return img.resize([width,height], resample=im.BICUBIC)
# a = im.open('5.png')
# a = downscale(a, 4)

# for data in glob.glob('data/train/*.png'):
#     a = im.open(data)
#     a = downscale(a, factor=2)
#     a.save(f'data/small/{data[-10:]}')


def preprocess(img_dir, width=args.shape[0], height=args.shape[1]):
    y = im.open(img_dir)
    y = y.convert('RGB')

    y = y.resize((width,height), resample=im.BICUBIC)

    factor = random_multiplier()
    x = color_shift(y, factor)
    y = color_shift(y, m_true)

    x = np.asarray(x, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)

    return x,y


def gen_data(datas, mode=color_shift, batch_size=1):
    while True:
        ix = np.random.choice(np.arange(len(datas)), batch_size)
        x_list = []
        y_list = []
        for i in ix:
            x,y = preprocess(datas[i])
            
            x_list.append(x)
            y_list.append(y)

        y_list = np.array(y_list)
        y_list = vgg.predict_on_batch(y_list)

        yield np.array(x_list), np.array(y_list)

# gen = gen_data(x)
# x,y = next(gen)

# train_set , val_set = tts(data, test_size = 0.2)

def conv_lr(layer_x, filters, act=LeakyReLU, use_bias=False):
    y = Conv2D(filters, kernel_size=3 \
            , padding="same")(layer_x)
    # layer_x = act(layer_x)
    return y

def network():
    # Model
    input_layer = Input(shape=[height,width,channels])
    res = [input_layer]

    #D1
    d1 = conv_lr(input_layer,48)
    d1 = conv_lr(d1,48)
    d1 = MaxPool2D()(d1)
    res.append(d1)

    #D2
    d2 = conv_lr(d1,48)
    d2 = MaxPool2D()(d2)
    res.append(d2)

    #D3
    d3 = conv_lr(d2,48)
    d3 = MaxPool2D()(d3)
    res.append(d3)

    #D4
    d4 = conv_lr(d3,48)
    d4 = MaxPool2D()(d4)
    res.append(d4)

    #Mid
    mid = conv_lr(d4,48)
    mid = MaxPool2D()(mid)
    mid = conv_lr(mid,48)
    #------------------------------------------------
    #U1
    u1 = UpSampling2D()(mid)
    u1 = Concatenate(axis=3)([res.pop(), u1])
    u1 = conv_lr(u1,96)
    u1 = conv_lr(u1,96)

    #U2
    u2 = UpSampling2D()(u1)
    u2 = Concatenate(axis=3)([res.pop(), u2])
    u2 = conv_lr(u2,96)
    u2 = conv_lr(u2,96)

    #U3
    u3 = UpSampling2D()(u2)
    u3 = Concatenate(axis=3)([res.pop(), u3])
    u3 = conv_lr(u3,96)
    u3 = conv_lr(u3,96)

    #U4
    u4 = UpSampling2D()(u3)
    u4 = Concatenate(axis=3)([res.pop(), u4])
    u4 = conv_lr(u4,96)
    u4 = conv_lr(u4,96)

    #U5
    u5 = UpSampling2D()(u4)
    u5 = Concatenate(axis=3)([res.pop(), u5])
    u5 = conv_lr(u5,64)
    u5 = conv_lr(u5,32)
    
    out = conv_lr(u5,channels,act=ReLU,use_bias=True)

    return Model(input_layer, out)

def vgg_net():
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]

    img = Input(shape=[height,width,channels])
    img_features = vgg(img)

    return Model(img, img_features)

vgg = vgg_net()
vgg.trainable = False
vgg.compile(loss='mse',
    optimizer=Adam(0.0002, 0.5),
    metrics=['accuracy'])

unet = network()

bad_image = Input(shape=[height,width,channels])
good_image = Input(shape=[height,width,channels])

gen_img = unet(bad_image)
bad_features = vgg(gen_img)

model =  Model(bad_image,bad_features)

def train():
    gen = gen_data(glob.glob('data/small/*.png'), batch_size=5)
    xs,ys = next(gen)

    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.99\
            , epsilon=1e-8, clipnorm=10.), loss=l2)

    filepath = f"dim{args.shape[0]}x{args.shape[1]}_s{args.steps}"\
                "e{epoch:02d}_loss[{loss:.2f}].h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss',save_best_only=True)

    model.summary()
    model.fit_generator(gen, steps_per_epoch=args.steps\
                        , epochs=args.epoch, callbacks=[checkpoint])


def test():
    model.load_weights(args.model)
    # model.summary()
    # model.get_layer('layer_name') also works
    # unet_layer_model = Model(inputs=model.input, outputs=model.get_layer("model_2").output)
    
    # model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.99, epsilon=1e-8\
    #             , clipnorm=10.), loss=l2)

    x,y = preprocess(args.image)

    # name = (args.image).split("/")[2].split(".")[0]
    name = ""
    imsave(name+'_x.png',x)

    if args.mult[0] != 1:
        imsave(name+'_truth.png',y)

    x = np.reshape(x,[1,args.shape[1],args.shape[0],channels])
    image = np.array(unet.predict(x))[0]
    imsave(name+'_y.png', image)

if __name__ == "__main__":
    if args.type == "train":
        train()
    else:
        test()