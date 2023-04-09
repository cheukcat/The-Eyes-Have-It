#!bin/sh
CONFIG=$1
WORKDIR=$2

shift 2
export CUDA_VISIBLE_DEVICES=0
python ./tools/val.py --py-config $CONFIG --work-dir $WORKDIR "$@"
