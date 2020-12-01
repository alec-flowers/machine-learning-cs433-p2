#!/bin/bash

ABSPATH=$(readlink -f $0)
ABSDIR=$(dirname $ABSPATH)
echo $ABSDIR

autopep8 -r $ABSDIR/../. --in-place --exclude venv
autopep8 -r $ABSDIR/../. --diff --exclude venv
python -m isort $ABSDIR/../.