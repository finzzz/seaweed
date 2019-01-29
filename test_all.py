import glob
import os
import subprocess

base_unet = "base_unetL2/dim128x96_l2_s180e118_loss[48.60].h5"
base_unetvgg = "base_unetVGG/[0.7-1.5]dim128x96_s180e119_vggloss[5377.98](unetvgg).h5"  # noqa

prog = "/Volumes/Toshiba/ML Models/"
prog_unet = prog + "unetL2 progressive/[0.7-1.1]dim128x96_l2_s180e92_loss[14.45].h5"  # noqa
prog_unetvgg = prog + "unetvgg progressive/[0.7-1.1]dim128x96_s180e86_loss[1520.37].h5"  # noqa

test_dir = "seaweed_test_img/"
img_list = glob.glob(f"{test_dir}*.jpg")

result_dir = "res/"

try:
    os.mkdir(result_dir)
except FileExistsError:
    pass

cmd_0 = ("python",)
cmd_1 = "test", "-mu", "0.1", "--alias"
cmd_2 = "--dir", result_dir, "-mod"

cmd_unet = cmd_0 + ("unet.py",) + cmd_1
cmd_unet_vgg = cmd_0 + ("unet~vgg.py",) + cmd_1
for img in img_list:
    # base_unet = bu
    temp = cmd_unet + ("bu",) + cmd_2 + (base_unet, "-i", img)
    subprocess.run(temp)

    # prog_unet = pu
    temp = cmd_unet + ("pu",) + cmd_2 + (prog_unet, "-i", img)
    subprocess.run(temp)

    # # base_unet_vgg = bv
    temp = cmd_unet_vgg + ("bv",) + cmd_2 + (base_unetvgg, "-i", img)
    subprocess.run(temp)

    # # prog_unet_vgg = pv
    temp = cmd_unet_vgg + ("pv",) + cmd_2 + (prog_unetvgg, "-i", img)
    subprocess.run(temp)
