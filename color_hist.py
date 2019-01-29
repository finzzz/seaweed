import argparse
import os.path as path
import sys
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt  # noqa : E402


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', nargs=2, help="true fake")
parser.add_argument('-c', '--channel', choices=["r", "g", "b", "all"],
                    default="all")
args = parser.parse_args()

try:
    for _ in args.image:
        filename = path.basename(_)
        dirname = path.dirname(_)
        if " " in filename or " " in dirname:
            raise Exception("file/dir name must not contain space")
except Exception as e:
    print(f"{e}, exitting...")
    sys.exit()

# "res 70%/1_truth.png"
true = cv2.imread(args.image[0])
fake = cv2.imread(args.image[1])
color_dict = {"b": 'turquoise', "g": 'lightgreen', "r": 'tomato'}
fake_color_dict = {"b": "indigo", "g": "green", "r": "magenta"}
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

for i, col in enumerate(fake_color):
    histr = cv2.calcHist([fake], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()
