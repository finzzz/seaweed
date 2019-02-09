import argparse
import os.path as path
import sys
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt  # noqa : E402


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help="true fake")
parser.add_argument('-c', '--channel', choices=["r", "g", "b", "all"],
                    default="all")
args = parser.parse_args()

try:
    filename = path.basename(args.image)
    dirname = path.dirname(args.image)
    if " " in filename or " " in dirname:
        raise Exception("file/dir name must not contain space")
except Exception as e:
    print(f"{e}, exitting...")
    sys.exit()

num = filename[2]
true_img = path.join(dirname, f"{num}_truth.png")
input_img = path.join(dirname, f"{num}_x.png")

# "res 70%/1_truth.png"
true = cv2.imread(true_img)
fake = cv2.imread(args.image)
inp = cv2.imread(input_img)
color_dict = {"b": 'turquoise', "g": 'lightgreen', "r": 'tomato'}
color_dict = {"b": 'darkblue', "g": 'darkgreen', "r": 'crimson'}
fake_color_dict = {"b": "indigo", "g": "green", "r": "red"}
if args.channel == "all":
    color = []
    fake_color = []
    for _ in color_dict.keys():
        color.append(color_dict.get(_))
        fake_color.append(fake_color_dict.get(_))
else:
    color = [color_dict.get(args.channel)]
    fake_color = [fake_color_dict.get(args.channel)]

plt.figure()
for i, col in enumerate(color):
    histr = cv2.calcHist([true], [i], None, [256], [0, 256])
    plt.plot(histr, color=col, linestyle='--')
    plt.xlim([0, 256])

for i, col in enumerate(color):
    histr = cv2.calcHist([inp], [i], None, [256], [0, 256])
    plt.plot(histr, color=col, linestyle='dotted')
    plt.xlim([0, 256])

for i, col in enumerate(fake_color):
    histr = cv2.calcHist([fake], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])

plt.show()
