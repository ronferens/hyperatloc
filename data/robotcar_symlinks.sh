#!/bin/bash

set -x
# change the directory
ROBOTCAR_SDK_ROOT=/home/dev/Software/robotcar-dataset-sdk

ln -s ${ROBOTCAR_SDK_ROOT}/models/ /home/dev/git/hyperatloc/data/robotcar_camera_models
ln -s ${ROBOTCAR_SDK_ROOT}/python/ /home/dev/git/hyperatloc/data/robotcar_sdk
set +x