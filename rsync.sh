#!/bin/zsh -e
# File: rsync.sh
# Date: Wed May 07 22:10:25 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
source ~/.aliasrc

if [[ "$1" == "go" ]]; then
	rsync -rlv --progress --partial --size-only ./ fit:~/DL/
elif [[ "$1" == "come" ]]; then
	rsync -rlv --progress --partial --size-only fit:~/DL/ ./
fi
