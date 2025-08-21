import torch
import torch.nn as nn
import os
from collections import OrderedDict
import argparse

# 导入OSNet模型定义
from OSNet import osnet_x0_25

def load_pretrained_weights(model, weight_path):
    """
    加载预训练权重（从engine.py中提取的简化版本）
    """
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # 移除module.前缀

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        print(f'警告: 无法加载预训练权重 "{weight_path}"')
    else:
        print(f'成功从 "{weight_path}" 加载预训练权重')
        if len(discarded_layers) > 0:
            print(f'** 以下层因不匹配的键或层大小而被丢弃: {discarded_layers}')

def convert_osnet_to_onnx(model_path, output_path, batch_size=1, input_size=(128, 64)):
    """
    将OSNet模型转换为ONNX格式
    
    参数:
        model_path: .pth模型文件路径
        output_path: 输出ONNX文件路径
        batch_size: 固定batch大小
        input_size: 输入图像尺寸 (width, height)
    """
    # 创建模型实例
    model = osnet_x0_25(num_classes=1, pretrained=False)
    
    # 加载预训练权重
    load_pretrained_weights(model, model_path)
    
    # 设置为评估模式
    model.eval()
    
    # 创建示例输入
    height, width = input_size[1], input_size[0]
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # 使用较稳定的opset版本
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,  # 不使用动态轴，固定batch和尺寸
        verbose=False
    )
    
    print(f"模型已成功导出到: {output_path}")
    print(f"输入尺寸: {batch_size} x 3 x {height} x {width}")
    
    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型验证成功!")
    except ImportError:
        print("ONNX包未安装，跳过模型验证")
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='将OSNet模型从.pth转换为ONNX格式')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='输入.pth模型文件路径')
    parser.add_argument('--output_path', type=str, default='osnet_x0_25.onnx',
                       help='输出ONNX文件路径')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='固定batch大小')
    parser.add_argument('--width', type=int, default=64,
                       help='输入图像宽度')
    parser.add_argument('--height', type=int, default=128,
                       help='输入图像高度')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 '{args.model_path}' 不存在")
        return
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # 执行转换
    convert_osnet_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        input_size=(args.width, args.height)
    )

if __name__ == "__main__":
    main()