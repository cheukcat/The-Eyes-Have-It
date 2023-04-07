#!bin/sh
CONFIG=$1
WORKDIR=$2
TIMESTAMP=`date "+%Y%m%d_%H%M%S"`

shift 2

python ./tools/train.py --py-config $CONFIG --work-dir $WORKDIR "$@" --timestamp $TIMESTAMP
