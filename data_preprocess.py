import os
import cv2
import numpy as np
from glob import glob

def preprocess_calibration_data(src_dir, dst_dir, target_size=(64, 128)):
    """
    预处理校准数据，转换为模型期望的NCHW格式
    
    参数:
        src_dir: 源图像目录
        dst_dir: 输出目录
        target_size: 目标尺寸 (width, height)
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    # ImageNet标准均值和标准差
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    image_files = glob(os.path.join(src_dir, '*.jpg')) + glob(os.path.join(src_dir, '*.png')) + glob(os.path.join(src_dir, '*.jpeg'))
    
    for img_path in image_files:
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 调整大小
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # BGR转RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 转换为float32
        img_float = img_rgb.astype(np.float32)
        
        # 标准化 (减去均值，除以标准差)
        img_normalized = (img_float - mean) / std
        
        # 转换为NCHW格式 [1, 3, H, W]
        img_nchw = np.transpose(img_normalized, (2, 0, 1))
        img_nchw = np.expand_dims(img_nchw, axis=0)
        
        # 保存为二进制文件
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(dst_dir, f"{base_name}.bin")
        
        img_nchw.tofile(output_path)
        print(f"Processed: {img_path} -> {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='预处理校准数据')
    parser.add_argument('--src_dir', type=str, required=True, help='源图像目录')
    parser.add_argument('--dst_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--width', type=int, default=64, help='目标宽度')
    parser.add_argument('--height', type=int, default=128, help='目标高度')
    
    args = parser.parse_args()
    
    preprocess_calibration_data(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        target_size=(args.width, args.height)
    )