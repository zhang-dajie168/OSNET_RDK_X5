# prepare_calibration_data.py
import cv2
import os
import numpy as np

def prepare_calibration_data(input_dir, output_dir, target_size=(256, 128)):
    """准备正确尺寸的校准数据"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 如果输入目录有图像，调整尺寸
    if os.path.exists(input_dir) and any(f.endswith(('.jpg', '.png', '.jpeg')) for f in os.listdir(input_dir)):
        for img_file in os.listdir(input_dir):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(input_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(img, (target_size[1], target_size[0]))
                    cv2.imwrite(os.path.join(output_dir, img_file), img_resized)
        print(f"Resized images from {input_dir} to {output_dir}")
    else:
        # 创建随机图像
        for i in range(100):
            img = np.random.randint(0, 255, (target_size[0], target_size[1], 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(output_dir, f'cal_{i:04d}.jpg'), img)
        print(f"Created 100 random calibration images in {output_dir}")

# 使用
prepare_calibration_data('/home/peng/视频/horizon_x5_open_explorer_v1.2.8-py310_20240926/yolov8_weights/OSnet/reid_image_jpg', './reid_image', (256, 128))
