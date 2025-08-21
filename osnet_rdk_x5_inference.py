# osnet_rdk_x5_inference_fixed.py
import cv2
import numpy as np
import os
import time
from hobot_dnn import pyeasy_dnn as dnn

class OSNetRDKX5Inference:
    def __init__(self, bin_model_path, input_size=(64, 128)):
        """
        初始化OSNet RDK X5推理器（修复版本）
        
        参数:
            bin_model_path: 量化后的.bin模型文件路径
            input_size: 输入图像尺寸 (width, height)
        """
        self.input_size = input_size
        
        # 检查模型文件是否存在
        if not os.path.exists(bin_model_path):
            raise FileNotFoundError(f"模型文件不存在: {bin_model_path}")
        
        # 加载模型
        start_time = time.time()
        try:
            self.models = dnn.load(bin_model_path)
            self.model = self.models[0]  # 获取第一个模型
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
        
        print(f"RDK X5模型加载成功: {bin_model_path}")
        print(f"模型加载耗时: {(time.time() - start_time)*1000:.2f}ms")
        
        # 获取模型信息
        try:
            self.input_tensor = self.model.inputs[0]
            self.output_tensor = self.model.outputs[0]
            
            input_shape = self.input_tensor.properties.shape
            output_shape = self.output_tensor.properties.shape
            
            print(f"输入形状: {input_shape}")
            print(f"输出形状: {output_shape}")
            print(f"输入数据类型: {self.input_tensor.properties.dtype}")
            print(f"输出数据类型: {self.output_tensor.properties.dtype}")
            
        except Exception as e:
            print(f"获取模型信息失败: {e}")
            # 设置默认值
            input_shape = [1, 3, 128, 64]  # NCHW格式
            output_shape = [1, 512]
        
        # 根据模型量化文档，使用正确的预处理参数
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([0.01712475, 0.017507, 0.01742919], dtype=np.float32)
        
        # 性能统计
        self.total_preprocess_time = 0
        self.total_inference_time = 0
        self.process_count = 0
    
    def bgr2nv12(self, image):
        """
        将BGR图像转换为NV12格式（修复版本）
        """
        # 调整大小
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # 转换为YUV_I420
        yuv_i420 = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)
        
        height, width = resized.shape[:2]
        
        # YUV_I420格式：YYYYYYYY UU VV
        y_size = width * height
        u_size = y_size // 4
        v_size = y_size // 4
        
        # 提取Y分量
        y = yuv_i420[:y_size].reshape(height, width)
        
        # 提取U和V分量
        u = yuv_i420[y_size:y_size + u_size].reshape(height // 2, width // 2)
        v = yuv_i420[y_size + u_size:y_size + u_size + v_size].reshape(height // 2, width // 2)
        
        # 创建NV12格式：YYYYYYYY UVUV
        nv12 = np.zeros((height * 3 // 2, width), dtype=np.uint8)
        
        # Y平面
        nv12[:height, :] = y
        
        # UV平面（交错排列）
        uv_plane = nv12[height:, :]
        uv_plane = uv_plane.reshape(height // 2, width, 1)
        
        # 交错排列U和V分量
        for i in range(height // 2):
            for j in range(width // 2):
                uv_plane[i, j * 2, 0] = u[i, j]      # U
                uv_plane[i, j * 2 + 1, 0] = v[i, j]  # V
        
        return nv12
    
    def bgr2nv12_simple(self, image):
        """
        更简单的BGR到NV12转换方法
        """
        # 调整大小
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # 使用OpenCV直接转换到YUV，然后手动创建NV12
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
        height, width = yuv.shape[:2]
        
        # 分离Y、U、V通道
        y = yuv[:, :, 0]
        u = yuv[:, :, 1]
        v = yuv[:, :, 2]
        
        # 下采样U和V通道到一半分辨率
        u_half = cv2.resize(u, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        v_half = cv2.resize(v, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        
        # 创建NV12数据
        nv12 = np.zeros((height * 3 // 2, width), dtype=np.uint8)
        
        # Y平面
        nv12[:height, :] = y
        
        # UV平面（交错排列）
        uv_plane = nv12[height:, :].reshape(height // 2, width // 2, 2)
        
        # 交错排列U和V分量
        uv_plane[:, :, 0] = u_half  # U分量
        uv_plane[:, :, 1] = v_half  # V分量
        
        return nv12
    
    def bgr2nv12_correct(self, image):
        """
        正确的BGR到NV12转换方法
        """
        # 调整大小
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        height, width = resized.shape[:2]
        
        # 转换为YUV
        yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
        
        # 创建NV12数据
        nv12 = np.zeros((height * 3 // 2, width), dtype=np.uint8)
        
        # Y平面
        nv12[:height, :] = yuv[:, :, 0]
        
        # UV平面 - 需要下采样并交错
        # 首先下采样U和V到一半分辨率
        u = cv2.resize(yuv[:, :, 1], (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(yuv[:, :, 2], (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        
        # 交错U和V分量
        uv_plane = nv12[height:, :].reshape(-1)
        for i in range((height // 2) * (width // 2)):
            uv_plane[i * 2] = u.flat[i]
            uv_plane[i * 2 + 1] = v.flat[i]
        
        return nv12
    
    def preprocess_image(self, image):
        """
        预处理单张图像为NV12格式
        
        参数:
            image: 输入图像 (H, W, C) BGR格式
            
        返回:
            processed: 预处理后的NV12图像数据
        """
        start_time = time.time()
        
        # 转换为NV12格式
        nv12 = self.bgr2nv12_correct(image)
        
        # 转换为模型期望的形状 (1, height*3//2, width)
        processed = np.expand_dims(nv12, axis=0)
        
        self.total_preprocess_time += (time.time() - start_time) * 1000
        return processed
    
    def preprocess_batch(self, images):
        """
        预处理批量图像为NV12格式
        
        参数:
            images: 图像列表，每个元素为(H, W, C) BGR格式
            
        返回:
            processed: 预处理后的NV12图像数据 (N, H*3//2, W)
        """
        processed_images = []
        for img in images:
            nv12 = self.bgr2nv12_correct(img)
            processed_images.append(nv12)
        
        return np.array(processed_images)
    
    def extract_features(self, image):
        """
        提取单张图像的特征
        
        参数:
            image: 输入图像 (H, W, C) BGR格式
            
        返回:
            features: 特征向量 (512,)
        """
        try:
            # 预处理为NV12格式
            input_tensor = self.preprocess_image(image)
            
            # 打印输入张量信息用于调试
            print(f"输入张量形状: {input_tensor.shape}")
            print(f"输入张量数据类型: {input_tensor.dtype}")
            print(f"输入张量范围: [{input_tensor.min()}, {input_tensor.max()}]")
            
            # 推理
            start_time = time.time()
            outputs = self.model.forward(input_tensor)
            self.total_inference_time += (time.time() - start_time) * 1000
            self.process_count += 1
            
            # 提取特征
            if hasattr(outputs[0], 'buffer'):
                features = outputs[0].buffer
            else:
                features = outputs[0]
                
            features = np.squeeze(features)  # 去除batch和空间维度
            if features.ndim > 1:
                features = features.flatten()
            
            return features
            
        except Exception as e:
            print(f"特征提取错误: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_features_batch(self, images):
        """
        提取批量图像的特征
        
        参数:
            images: 图像列表，每个元素为(H, W, C) BGR格式
            
        返回:
            features: 特征矩阵 (N, 512)
        """
        try:
            # 预处理为NV12格式
            input_tensor = self.preprocess_batch(images)
            
            # 推理
            start_time = time.time()
            outputs = self.model.forward(input_tensor)
            self.total_inference_time += (time.time() - start_time) * 1000
            self.process_count += len(images)
            
            # 提取特征
            if hasattr(outputs[0], 'buffer'):
                features = outputs[0].buffer
            else:
                features = outputs[0]
            
            # 处理输出形状 (N, 512, 1, 1) -> (N, 512)
            if features.ndim == 4:
                features = features.reshape(features.shape[0], features.shape[1])
            
            return features
            
        except Exception as e:
            print(f"批量特征提取错误: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # 其他方法保持不变...
    def extract_features_from_bboxes(self, bboxes, ori_img, input_format='tlwh'):
        """
        从原始图像中裁剪多个bbox区域并提取特征
        
        参数:
            bboxes: 边界框数组 (N, 4) [x1, y1, w, h] 或 [x1, y1, x2, y2]
            ori_img: 原始图像 (H, W, C) BGR格式
            input_format: 边界框格式 ('tlwh' 或 'tlbr')
            
        返回:
            features: 特征矩阵 (N, 512)
            crops: 裁剪后的图像列表
        """
        bboxes = bboxes.copy()
        img_h, img_w = ori_img.shape[:2]
        
        # 转换bbox格式为xyxy
        if input_format == 'tlwh':
            bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
            bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        
        crops = []
        for box in bboxes:
            x1, y1, x2, y2 = box.round().astype('int')
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
            
            # 裁剪图像
            if x2 > x1 and y2 > y1:  # 确保有效的裁剪区域
                crop = ori_img[y1:y2, x1:x2]
                crops.append(crop)
        
        if not crops:
            return np.array([]), []
        
        # 提取特征
        features = self.extract_features_batch(crops)
        
        return features, crops
    
    def compute_similarity(self, features1, features2):
        """
        计算两个特征向量之间的余弦相似度
        
        参数:
            features1: 特征向量1 (512,) 或 (N, 512)
            features2: 特征向量2 (512,) 或 (N, 512)
            
        返回:
            similarity: 余弦相似度
        """
        if features1 is None or features2 is None:
            return 0.0
            
        # 归一化特征向量
        if len(features1.shape) == 1:
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 > 0 and norm2 > 0:
                features1 = features1 / norm1
                features2 = features2 / norm2
                similarity = np.dot(features1, features2)
            else:
                similarity = 0.0
        else:
            norm1 = np.linalg.norm(features1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(features2, axis=1, keepdims=True)
            features1 = features1 / np.where(norm1 > 0, norm1, 1)
            features2 = features2 / np.where(norm2 > 0, norm2, 1)
            similarity = np.dot(features1, features2.T)
        
        return similarity
    
    def get_performance_stats(self):
        """
        获取性能统计
        """
        if self.process_count == 0:
            return {
                'avg_preprocess_time': 0,
                'avg_inference_time': 0,
                'avg_total_time': 0,
                'total_process_count': 0
            }
        
        avg_preprocess = self.total_preprocess_time / self.process_count
        avg_inference = self.total_inference_time / self.process_count
        
        return {
            'avg_preprocess_time': avg_preprocess,
            'avg_inference_time': avg_inference,
            'avg_total_time': avg_preprocess + avg_inference,
            'total_process_count': self.process_count
        }

# 使用示例
if __name__ == "__main__":
    # 初始化RDK X5推理器
    rdk_inference = OSNetRDKX5Inference(
        bin_model_path="osnet_64x128_nv12.bin",  # 量化后的bin模型
        input_size=(64, 128)  # (width, height)
    )
    
    # 示例1: 提取单张图像特征
    image1 = cv2.imread("./test_image/0001_c5_f0051487.jpg")
    if image1 is not None:
        print(f"原始图像形状: {image1.shape}")
        features1 = rdk_inference.extract_features(image1)
        print(f"特征向量形状: {features1.shape}")
        print(f"特征向量范数: {np.linalg.norm(features1)}")
        print(f"特征数据类型: {features1.dtype}")
    else:
        print("无法加载图像1")
    
    image2 = cv2.imread("./test_image/0002_c1_f0053158.jpg")
    if image2 is not None:
        features2 = rdk_inference.extract_features(image2)
        print(f"特征向量形状: {features2.shape}")
        print(f"特征向量范数: {np.linalg.norm(features2)}")
    else:
        print("无法加载图像2")
    
    # 示例2: 计算相似度
    if image1 is not None and image2 is not None:
        similarity = rdk_inference.compute_similarity(features1, features2)
        print(f"相似度: {similarity:.4f}")
    
    # 输出性能统计
    stats = rdk_inference.get_performance_stats()
    print(f"\n性能统计:")
    print(f"平均预处理时间: {stats['avg_preprocess_time']:.2f}ms")
    print(f"平均推理时间: {stats['avg_inference_time']:.2f}ms")
    print(f"平均总时间: {stats['avg_total_time']:.2f}ms")
    print(f"总处理次数: {stats['total_process_count']}")
