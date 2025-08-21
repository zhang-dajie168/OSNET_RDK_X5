import cv2
import numpy as np
import onnxruntime as ort
import torch

class OSNetONNXInference:
    def __init__(self, onnx_model_path, device='cpu', input_size=(64, 128)):
        """
        初始化OSNet ONNX推理器
        
        参数:
            onnx_model_path: ONNX模型文件路径
            device: 推理设备 ('cpu' 或 'cuda')
            input_size: 输入图像尺寸 (width, height)
        """
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        # 配置ONNX Runtime
        providers = ['CPUExecutionProvider']
        if device.lower() == 'cuda' and ort.get_device() == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 创建推理会话
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"ONNX模型加载成功: {onnx_model_path}")
        print(f"输入名称: {self.input_name}, 输出名称: {self.output_name}")
        print(f"输入形状: {self.session.get_inputs()[0].shape}")
        print(f"输出形状: {self.session.get_outputs()[0].shape}")
    
    def preprocess_image(self, image):
        """
        预处理单张图像，与engine.py中的crop_and_resize保持一致
        
        参数:
            image: 输入图像 (H, W, C) BGR格式
            
        返回:
            processed: 预处理后的图像张量 (1, 3, H, W)
        """
        # 调整大小
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # BGR转RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 转换为float32并归一化到[0,1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # 标准化 (与engine.py中的mean/std保持一致)
        normalized = (normalized - self.mean) / self.std
        
        # 转换为CHW格式
        chw = np.transpose(normalized, (2, 0, 1))
        
        # 添加batch维度
        batched = np.expand_dims(chw, axis=0)
        
        return batched
    
    def preprocess_batch(self, images):
        """
        预处理批量图像
        
        参数:
            images: 图像列表，每个元素为(H, W, C) BGR格式
            
        返回:
            processed: 预处理后的图像张量 (N, 3, H, W)
        """
        processed_images = []
        for img in images:
            # 调整大小
            resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
            
            # BGR转RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # 转换为float32并归一化到[0,1]
            normalized = rgb.astype(np.float32) / 255.0
            
            # 标准化
            normalized = (normalized - self.mean) / self.std
            
            # 转换为CHW格式
            chw = np.transpose(normalized, (2, 0, 1))
            
            processed_images.append(chw)
        
        return np.array(processed_images)
    
    def extract_features(self, image):
        """
        提取单张图像的特征
        
        参数:
            image: 输入图像 (H, W, C) BGR格式
            
        返回:
            features: 特征向量 (512,)
        """
        # 预处理
        input_tensor = self.preprocess_image(image)
        
        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # 提取特征
        features = outputs[0][0]  # 取第一个batch的结果
        
        return features
    
    def extract_features_batch(self, images):
        """
        提取批量图像的特征
        
        参数:
            images: 图像列表，每个元素为(H, W, C) BGR格式
            
        返回:
            features: 特征矩阵 (N, 512)
        """
        # 预处理
        input_tensor = self.preprocess_batch(images)
        
        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # 提取特征
        features = outputs[0]  # 所有batch的结果
        
        return features
    
    def extract_features_from_bboxes(self, bboxes, ori_img, input_format='tlwh'):
        """
        从原始图像中裁剪多个bbox区域并提取特征（类似engine.py中的crop_and_resize）
        
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
            crop = ori_img[y1:y2, x1:x2]
            crops.append(crop)
        
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
        # 归一化特征向量
        if len(features1.shape) == 1:
            features1 = features1 / np.linalg.norm(features1)
            features2 = features2 / np.linalg.norm(features2)
            similarity = np.dot(features1, features2)
        else:
            features1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
            features2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
            similarity = np.dot(features1, features2.T)
        
        return similarity

# 使用示例
if __name__ == "__main__":
    # 初始化推理器
    onnx_inference = OSNetONNXInference(
        onnx_model_path="osnet_x0_25.onnx",
        device="cpu",  # 或 "cuda"
        input_size=(64, 128)  # (width, height)
    )
    
    # 示例1: 提取单张图像特征
    image1 = cv2.imread("./reid_image/0002_c1_f0054854.jpg")
    if image1 is not None:
        features1 = onnx_inference.extract_features(image1)
        print(f"特征向量形状: {features1.shape}")
        print(f"特征向量范数: {np.linalg.norm(features1)}")
    
    image2 = cv2.imread("./reid_image/0002_c1_f0054118.jpg")
    if image2 is not None:
        features2 = onnx_inference.extract_features(image2)
        print(f"特征向量形状: {features2.shape}")
        print(f"特征向量范数: {np.linalg.norm(features2)}")
    
    # # 示例2: 从多个bbox提取特征（类似StrongSort中的用法）
    # bboxes = np.array([
    #     [100, 100, 50, 150],  # [x, y, w, h]
    #     [200, 120, 60, 160]
    # ])
    
    # features, cropped_images = onnx_inference.extract_features_from_bboxes(
    #     bboxes, image, input_format='tlwh'
    # )
    # print(f"批量特征形状: {features.shape}")
    
    # 示例3: 计算相似度
    # if len(features) >= 2:
    similarity = onnx_inference.compute_similarity(features1, features2)
    print(f"相似度: {similarity:.4f}")