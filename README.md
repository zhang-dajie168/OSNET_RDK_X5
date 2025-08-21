

## 模型下载 yolov8量化模型及osnet预处理模型及量化模型
    链接: https://pan.baidu.com/s/1r9b7VCQYQC7QwaUGiw617g 提取码: g58e


## export_osnet_to_onnx.py

    osnet_x0_25.pth ===>> osnet_x0_25.onnx


## 量化
    1.配置文件：OSNet_x0_25_config.yaml
    2.reid_image:256*128.jpg格式的行人数据
    3.reid_image_correct：reid_image转换nv12格式校准图片数据  ，运行 python data_preprocess.py 进行转换
    

## 部署推理 
    推送到板子端，路径配置，运行osnet_rdk_x5_inference.py
    
