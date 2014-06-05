#!/bin/bash -e
# File: run_paste.sh
# Date: Fri Jun 06 00:36:03 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
./paste_image.py -i 'convolved_layer0*.jpg' -o conv0-label"$1".jpg
./paste_image.py -i 'convolved_layer1*.jpg' -o conv1-label"$1".jpg
./paste_image.py -i 'convolved_layer2*.jpg' -o conv2-label"$1".jpg
