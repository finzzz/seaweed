  # noqa : F821
import numpy as np
import glob
import argparse
import os.path
import utils
import network
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from scipy.misc import imsave

# default loss:
l1 = "mean_absolute_error"
l2 = "mean_squared_error"

parser = argparse.ArgumentParser()
parser.add_argument("type", help="train/test/continue train model",
                    choices=["train", "test", "continue"])
parser.add_argument("-mod", "--model", help="model.h5 file")
parser.add_argument("-i", "--image", help="image to test")
parser.add_argument("-mu", "--mult", help="multiplier factor", nargs="+",
                    type=float, default=[0.5, 0.7, 0.8, 1, 1.2, 1.3, 1.5])
parser.add_argument("-mt", "--true_m", help="multiplier factor (true)",
                    type=float, default=1)
parser.add_argument("-sh", "--shape", help="image shape [width,height]",
                    type=int, nargs=2, default=[128, 96])
parser.add_argument("-l", "--loss", help="loss function, l1,l2,l1l2",
                    default=l2)
parser.add_argument("-st", "--steps", help="steps per epoch",
                    type=int, default=180)
parser.add_argument("-e", "--epoch", help="number of epoch",
                    type=int, default=3)
parser.add_argument("--dir", default="./")
parser.add_argument("--alias", default="")
args = parser.parse_args()

# channels
channels = 4

# shape
width = args.shape[0]
height = args.shape[1]
dims = {"width": args.shape[0],
        "height": args.shape[1]}

if args.loss == "l1":
    args.loss = l1

# alias
if args.loss == l1:
    alias = "l1"
elif args.loss == l2:
    alias = "l2"

if args.type == "test" and not args.model and not args.image:
    parser.error("test needs image & model")


def gen_data(datas, mode=utils.color_shift, batch_size=1):
    while True:
        ix = np.random.choice(np.arange(len(datas)), batch_size)
        x_list = []
        y_list = []
        for i in ix:
            x, y = utils.preprocess(datas[i], dims=dims, channels=channels,
                                    mult=args.mult, m_true=args.true_m)

            x_list.append(x)
            y_list.append(y)

        yield np.array(x_list), np.array(y_list)


def train(continue_flag=False):
    gen = gen_data(glob.glob('data/small/*.png'), batch_size=5)
    xs, ys = next(gen)

    if continue_flag:
        model = load_model(args.model)
    else:
        model = network.unet(height, width, channels)

    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.99,
                  epsilon=1e-8, clipnorm=10.), loss=args.loss)

    filepath = f"dim{width}x{height}_{alias}_s{args.steps}"\
               f"e{epoch:02d}_loss[{loss:.2f}].h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True)

    model.fit_generator(gen, steps_per_epoch=args.steps,
                        epochs=args.epoch, callbacks=[checkpoint])


def test(dir):
    model = load_model(args.model)

    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.99,
                  epsilon=1e-8, clipnorm=10.), loss=args.loss)

    x, y = utils.preprocess(args.image, dims=dims,
                            mult=args.mult, channels=channels)

    name = os.path.basename(args.image).split(".")[0]
    imsave(f"{dir}{name}_x.png", x)

    if args.mult[0] != 1:
        imsave(f"{dir}{name}_truth.png", y)

    x = np.reshape(x, [1, height, width, channels])
    y = np.reshape(y, [1, height, width, channels])
    prediction = model.predict(x)
    prediction_metrics = model.evaluate(x=x, y=y, verbose=0)
    imsave(f"{dir}{args.alias}{name}_y_{prediction_metrics:0.2f}.png",
           np.array(prediction)[0])


if __name__ == "__main__":
    if args.type == "train":
        train()
    elif args.type == "test":
        test(dir=args.dir)
    elif args.type == "continue":
        train(continue_flag=True)
