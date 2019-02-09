import matplotlib
matplotlib.use('TkAgg')
import numpy as np # noqa:E402
from matplotlib import pyplot as plt # noqa:E402
from scipy import misc # noqa:E402
import scipy.stats as stats # noqa:E402


# line style = ":", "-.","--"
def densityplot(img_path, color, style="-"):
    img = np.asarray(misc.imread(img_path))
    img_hsv = matplotlib.colors.rgb_to_hsv(img[..., :3])
    lu = img_hsv[..., 1].flatten()
    density = stats.gaussian_kde(lu)
    n, x, _ = plt.hist(lu, 256, histtype='step', density=True)
    plt.plot(density(x), color=color, linestyle=style)


plt.figure()
densityplot("res_70%/3_truth.png", color="k", style="-.")
densityplot("res_70%/3_x.png", color="k")
densityplot("res_70%/bu3_y_72.97.png", "red")
densityplot("res_70%/bv3_y_194.18.png", "orange")
densityplot("res_70%/pu3_y_14.70.png", "green")
densityplot("res_70%/pv3_y_71.08.png", "blue")
plt.xlim(0.5, 256)
plt.show()
