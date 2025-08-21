#!/usr/bin/env bash
set -e -v

cd $(dirname $0) || exit

config_file="./OSNet_x0_25_config.yaml"
model_type="onnx"

# 构建模型
hb_mapper makertbin --config ${config_file} \
                    --model-type ${model_type}
