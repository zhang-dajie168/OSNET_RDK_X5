#!/usr/bin/env bash
set -e -v

cd $(dirname $0) || exit

python3 data_preprocess.py \
  --src_dir ./reid_images \          # 原始图像目录
  --dst_dir ./reid_image_correct \   # 处理后的校准数据目录
  --width 64 \
  --height 128
