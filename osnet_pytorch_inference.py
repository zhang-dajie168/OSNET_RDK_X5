# osnet_pytorch_inference_optimized.py
import numpy as np
import cv2
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from collections import OrderedDict
import pickle
from functools import partial

# 导入OSNet模型定义
import sys
sys.path.append(os.path.dirname(__file__))

from OSNet import osnet_x0_25

class OSNetPyTorchInference:
    def __init__(self, model_path, device='cuda'):
        """
        OSNet PyTorch推理器 - 优化版本
        Args:
            model_path: osnet_x0_25.pth 模型路径
            device: 运行设备 ('cuda', 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型 - 使用与engine.py一致的方式
        start_time = time.time()
        self.model = self._load_model_engine_style(model_path)
        model_load_time = (time.time() - start_time) * 1000
        print(f"模型加载耗时: {model_load_time:.2f}ms")
        
        # 使用与engine.py完全一致的预处理参数
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),  # 与engine.py的crop_and_resize一致
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 模型设置为评估模式
        self.model.eval()
        print(f"模型加载成功: {model_path}")

    def _load_checkpoint(self, fpath):
        """与engine.py一致的checkpoint加载方式"""
        if fpath is None:
            raise ValueError('File path is None')
        fpath = os.path.abspath(os.path.expanduser(fpath))
        if not os.path.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))
        map_location = None if torch.cuda.is_available() else 'cpu'
        try:
            checkpoint = torch.load(fpath, map_location=map_location)
        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )
        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise
        return checkpoint

    def _load_model_engine_style(self, model_path):
        """使用与engine.py完全一致的模型加载方式"""
        # 创建标准OSNet模型
        model = osnet_x0_25(num_classes=1, pretrained=False)
        
        if os.path.exists(model_path):
            checkpoint = self._load_checkpoint(model_path)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            model_dict = model.state_dict()
            new_state_dict = OrderedDict()
            matched_layers, discarded_layers = [], []

            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # discard module.

                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
                    matched_layers.append(k)
                else:
                    discarded_layers.append(k)

            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)

            if len(matched_layers) == 0:
                print(
                    '警告: 预训练权重"{}"无法加载，请手动检查键名'.format(model_path)
                )
            else:
                print('成功加载预训练权重从"{}"'.format(model_path))
                if len(discarded_layers) > 0:
                    print('** 以下层因不匹配的键或层大小而被丢弃: {}'.format(discarded_layers))
        else:
            print("警告: 未找到预训练权重，使用随机初始化")
        
        # 移除分类头，只保留特征提取部分 - 与StrongSORT一致
        model.classifier = nn.Identity()
        
        return model.to(self.device)

    def preprocess_image(self, image_path):
        """
        预处理图像 - 与engine.py的crop_and_resize逻辑一致
        """
        # 使用PIL读取图像
        img = Image.open(image_path).convert('RGB')
        
        # 应用转换
        img_tensor = self.transform(img)
        
        # 添加batch维度
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)

    def preprocess_cv2_image(self, cv2_img):
        """
        处理OpenCV图像 - 与engine.py的crop_and_resize逻辑一致
        """
        # 转换BGR到RGB
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # 应用转换
        img_tensor = self.transform(img_pil)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)

    def extract_features(self, image_path):
        """
        提取特征向量 - 添加特征归一化
        """
        with torch.no_grad():
            # 预处理
            input_tensor = self.preprocess_image(image_path)
            
            # 前向传播
            features = self.model(input_tensor)
            
            # 特征归一化 - 与StrongSORT一致
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            # 转换为numpy数组
            features = features.cpu().numpy().squeeze()
            
        return features

    def extract_features_from_cv2(self, cv2_img):
        """
        从OpenCV图像提取特征
        """
        with torch.no_grad():
            input_tensor = self.preprocess_cv2_image(cv2_img)
            features = self.model(input_tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            features = features.cpu().numpy().squeeze()
        return features

    def extract_features_batch(self, image_paths):
        """
        批量提取特征
        """
        features_list = []
        
        for img_path in image_paths:
            features = self.extract_features(img_path)
            features_list.append(features)
            
        return np.array(features_list)

    def calculate_similarity(self, features1, features2):
        """
        计算余弦相似度 - 使用归一化后的特征
        """
        # 确保是1D数组
        features1 = features1.flatten()
        features2 = features2.flatten()
        
        # 计算余弦相似度
        similarity = 1 - cosine(features1, features2)
        return similarity

    def compare_images(self, image_path1, image_path2, threshold=0.75):
        """
        比较两张图像
        """
        print(f"\n{'='*60}")
        print(f"🔍 图像比较: {os.path.basename(image_path1)} vs {os.path.basename(image_path2)}")
        print(f"{'='*60}")
        
        # 提取特征
        start_time = time.time()
        feat1 = self.extract_features(image_path1)
        feat2 = self.extract_features(image_path2)
        extract_time = (time.time() - start_time) * 1000
        
        # 计算相似度
        start_time = time.time()
        similarity = self.calculate_similarity(feat1, feat2)
        similarity_time = (time.time() - start_time) * 1000
        
        print(f"特征提取耗时: {extract_time:.2f}ms")
        print(f"相似度计算耗时: {similarity_time:.2f}ms")
        print(f"📊 余弦相似度: {similarity:.6f}")
        print(f"💯 匹配分数: {similarity*100:.2f}%")
        
        # 分析结果
        predicted = 'same' if similarity > threshold else 'different'
        print(f"🔮 预测结果: {predicted} (阈值: {threshold})")
        
        self._analyze_similarity(similarity, threshold)
        
        return similarity, feat1, feat2, predicted

    def _analyze_similarity(self, similarity, threshold):
        """分析相似度结果"""
        if similarity > 0.9:
            print("✅ 极高相似度 - 极大概率是同一个人")
        elif similarity > threshold + 0.1:
            print("✅ 高相似度 - 很可能是同一个人")
        elif similarity > threshold:
            print("⚠️  中等相似度 - 需要进一步验证")
        elif similarity > threshold - 0.1:
            print("❓ 低相似度 - 可能不是同一个人")
        elif similarity > 0.3:
            print("❌ 极低相似度 - 很可能不是同一个人")
        else:
            print("❌ 极不相似 - 确定不是同一个人")

    def test_model_performance(self, test_cases, threshold=0.75):
        """
        测试模型性能
        test_cases: [(img1, img2, expected_label), ...]
        """
        results = []
        
        for i, (img1, img2, expected) in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"测试用例 {i+1}: {expected}")
            print(f"{'='*60}")
            
            if not all(os.path.exists(img) for img in [img1, img2]):
                print("❌ 文件不存在")
                continue
            
            try:
                similarity, _, _, predicted = self.compare_images(img1, img2, threshold)
                
                correct = predicted == expected
                
                results.append({
                    'case': i+1,
                    'image1': os.path.basename(img1),
                    'image2': os.path.basename(img2),
                    'similarity': similarity,
                    'predicted': predicted,
                    'expected': expected,
                    'correct': correct,
                    'threshold': threshold
                })
                
                status = "✅" if correct else "❌"
                print(f"{status} 预测: {predicted}, 预期: {expected}")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                results.append({'case': i+1, 'error': str(e)})
        
        # 输出性能报告
        self._print_performance_report(results)
        return results

    def find_optimal_threshold(self, test_cases):
        """寻找最佳相似度阈值"""
        thresholds = np.arange(0.5, 0.95, 0.05)
        best_accuracy = 0
        best_threshold = 0.75
        
        for threshold in thresholds:
            correct = 0
            total = 0
            
            for img1, img2, expected in test_cases:
                if not all(os.path.exists(img) for img in [img1, img2]):
                    continue
                
                feat1 = self.extract_features(img1)
                feat2 = self.extract_features(img2)
                similarity = self.calculate_similarity(feat1, feat2)
                predicted = 'same' if similarity > threshold else 'different'
                
                if predicted == expected:
                    correct += 1
                total += 1
            
            if total > 0:
                accuracy = correct / total
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
        
        print(f"最佳阈值: {best_threshold:.3f}, 准确率: {best_accuracy:.3f}")
        return best_threshold

    def _print_performance_report(self, results):
        """输出性能报告"""
        if not results:
            return
            
        correct_count = sum(1 for r in results if 'correct' in r and r['correct'])
        total_count = sum(1 for r in results if 'correct' in r)
        
        accuracy = correct_count / total_count * 100 if total_count > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"📊 性能报告")
        print(f"{'='*60}")
        print(f"总测试用例: {len(results)}")
        print(f"正确识别: {correct_count}/{total_count}")
        print(f"准确率: {accuracy:.1f}%")
        print(f"使用阈值: {results[0]['threshold'] if results else 'N/A'}")
        
        # 显示错误案例
        errors = [r for r in results if 'correct' in r and not r['correct']]
        if errors:
            print(f"\n❌ 错误案例:")
            for error in errors:
                print(f"  用例{error['case']}: {error['image1']} vs {error['image2']}")
                print(f"     相似度: {error['similarity']:.4f}, 预测: {error['predicted']}, 预期: {error['expected']}")

def main():
    # 模型路径
    model_path = "./osnet_x0_25.pth"
    
    # 测试图像目录
    test_dir = "./test_image/"
    
    # 测试用例
    test_cases = [
        (os.path.join(test_dir, "0001_c5_f0051487.jpg"), 
         os.path.join(test_dir, "0001_c5_f0051607.jpg"), "same"),
        
        (os.path.join(test_dir, "0001_c5_f0051487.jpg"), 
         os.path.join(test_dir, "0007_c2_f0047184.jpg"), "different"),
    ]
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    # 创建推理器
    print("初始化PyTorch推理引擎...")
    inference_engine = OSNetPyTorchInference(model_path)
    
    # 寻找最佳阈值
    print("\n寻找最佳阈值...")
    optimal_threshold = inference_engine.find_optimal_threshold(test_cases)
    
    # 使用最佳阈值测试模型性能
    print(f"\n使用最佳阈值 {optimal_threshold:.3f} 测试性能...")
    results = inference_engine.test_model_performance(test_cases, optimal_threshold)
    
    # 特征分析（可选）
    if input("\n是否进行特征分析？(y/n): ").lower() == 'y':
        image_paths = []
        labels = []
        
        for img1, img2, expected in test_cases:
            image_paths.extend([img1, img2])
            label1 = os.path.basename(img1).split('_')[0]
            label2 = os.path.basename(img2).split('_')[0]
            labels.extend([label1, label2])
        
        # 提取特征并分析
        features = inference_engine.extract_features_batch(image_paths)
        print(f"\n特征形状: {features.shape}")
        print(f"特征范围: [{features.min():.6f}, {features.max():.6f}]")
        print(f"特征均值: {features.mean():.6f}")
        print(f"特征标准差: {features.std():.6f}")

if __name__ == "__main__":
    main()