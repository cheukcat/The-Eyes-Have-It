#!bin/sh
CONFIG=$1
WORKDIR=$2

shift 2

python ./tools/train.py --py-config $CONFIG --work-dir $WORKDIR "$@"
