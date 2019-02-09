Installation:
1. install python 3.7
2. install pipenv  >> "pip install pipenv"
3. run "pipenv lock"
4. run "pipenv sync"


How to train:
1. enter pipenv >> "pipenv shell"
2. train model >> "python unet.py train -st 100 -e 3"
   -st is training steps
   -e is training epoch
3. more help >> "python unet.py -h"


How to test:
1. enter pipenv >> "pipenv shell"
2. test model >> "python unet.py test -i "image.png" -mu 0.7 -mod "model.h5" "
   -mu is to decrease saturation of test image, 0 = no color, 1 = no change
   -mod is path to model
   -i is image path
3. this will output "picturename_x.png" "picturename_y.png" "picturename_truth.png"
   x is image after preprocess
   y is output of model
   truth is image without preprocess


How to continue train:
1. enter pipenv >> "pipenv shell"
2. continue training >> "python unet.py continue -st 100 -e 3 -mod "previousmodel.h5" "


ps. data folder contains DIV2K dataset
