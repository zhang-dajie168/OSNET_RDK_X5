#include "OSNet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>

// 自定义文件系统函数（兼容 C++11）
bool directoryExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

std::vector<std::string> getImageFiles(const std::string& directory) {
    std::vector<std::string> image_files;
    std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        return image_files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) { // 常规文件
            std::string filename = entry->d_name;
            std::string extension;
            
            // 提取文件扩展名
            size_t dot_pos = filename.find_last_of(".");
            if (dot_pos != std::string::npos) {
                extension = filename.substr(dot_pos);
                // 转换为小写
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                // 检查是否是图片格式
                if (std::find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end()) {
                    image_files.push_back(directory + "/" + filename);
                }
            }
        }
    }
    closedir(dir);
    
    return image_files;
}

std::string getFilenameFromPath(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        return path.substr(last_slash + 1);
    }
    return path;
}

int main(int argc, char** argv) {
    // 初始化 RDK X5 推理器
    std::string model_path = "osnet_64x128_nv12.bin";
    cv::Size input_size(64, 128);
    
    try {
        OSNetRDKX5Inference rdk_inference(model_path, input_size);
        
        // 初始化模型
        if (!rdk_inference.init_model()) {
            std::cerr << "模型初始化失败" << std::endl;
            return -1;
        }
        
        // 预热
        cv::Mat warmup_img(256, 128, CV_8UC3);
        cv::randu(warmup_img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        for (int i = 0; i < 3; ++i) {
            auto features = rdk_inference.extract_features(warmup_img);
        }
        
        // 获取 test_image 目录下的所有图片文件
        std::string image_dir = "test_image";
        std::vector<std::string> image_paths;
        
        if (!directoryExists(image_dir)) {
            std::cerr << "目录不存在: " << image_dir << std::endl;
            return -1;
        }
        
        image_paths = getImageFiles(image_dir);
        
        if (image_paths.empty()) {
            std::cerr << "在目录 " << image_dir << " 中未找到图片文件" << std::endl;
            return -1;
        }
        
        // 按文件名排序
        std::sort(image_paths.begin(), image_paths.end());
        
        std::cout << "找到 " << image_paths.size() << " 张图片:" << std::endl;
        for (const auto& path : image_paths) {
            std::cout << "  " << getFilenameFromPath(path) << std::endl;
        }
        std::cout << std::endl;
        
        // 加载第一张图片作为基准
        std::string first_image_path = image_paths[0];
        cv::Mat first_image = cv::imread(first_image_path);
        
        if (first_image.empty()) {
            std::cerr << "无法加载第一张图像: " << first_image_path << std::endl;
            return -1;
        }
        
        std::cout << "基准图像: " << getFilenameFromPath(first_image_path) << std::endl;
        std::cout << "图像形状: " << first_image.cols << "x" << first_image.rows << "x" << first_image.channels() << std::endl;
        
        // 提取基准图像特征
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> base_features = rdk_inference.extract_features(first_image);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        double process_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / 1000.0;
        
        std::cout << "基准图像处理时间: " << process_time << "ms" << std::endl;
        std::cout << "特征向量大小: " << base_features.size() << std::endl;
        
        // 计算基准特征范数
        float base_norm = 0.0f;
        for (auto val : base_features) {
            base_norm += val * val;
        }
        base_norm = std::sqrt(base_norm);
        std::cout << "基准特征向量范数: " << base_norm << std::endl << std::endl;
        
        // 计算与其他所有图片的相似度
        std::vector<std::pair<std::string, float>> similarity_results;
        
        for (size_t i = 1; i < image_paths.size(); ++i) {
            std::string current_image_path = image_paths[i];
            std::string filename = getFilenameFromPath(current_image_path);
            
            try {
                cv::Mat current_image = cv::imread(current_image_path);
                if (current_image.empty()) {
                    std::cerr << "无法加载图像: " << current_image_path << ", 跳过..." << std::endl;
                    continue;
                }
                
                // 提取当前图像特征
                auto current_start = std::chrono::high_resolution_clock::now();
                std::vector<float> current_features = rdk_inference.extract_features(current_image);
                auto current_end = std::chrono::high_resolution_clock::now();
                
                double current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    current_end - current_start).count() / 1000.0;
                
                // 计算相似度
                float similarity = rdk_inference.compute_similarity(base_features, current_features);
                
                similarity_results.emplace_back(filename, similarity);
                
                std::cout << "[" << i << "/" << image_paths.size()-1 << "] "
                          << filename << " -> 相似度: " << similarity 
                          << " (处理时间: " << current_time << "ms)" << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "处理图像 " << filename << " 时出错: " << e.what() << ", 跳过..." << std::endl;
            }
        }
        
        // 按相似度排序
        std::sort(similarity_results.begin(), similarity_results.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // 打印排序后的结果
        std::cout << "\n=== 相似度排序结果 ===" << std::endl;
        std::cout << "基准图像: " << getFilenameFromPath(first_image_path) << std::endl;
        std::cout << "==========================================" << std::endl;
        
        for (size_t i = 0; i < similarity_results.size(); ++i) {
            std::cout << "排名 " << i+1 << ": " << similarity_results[i].first 
                      << " -> 相似度: " << similarity_results[i].second << std::endl;
        }
        
        // 输出统计信息
        if (!similarity_results.empty()) {
            float max_similarity = similarity_results[0].second;
            float min_similarity = similarity_results.back().second;
            float avg_similarity = 0.0f;
            
            for (const auto& result : similarity_results) {
                avg_similarity += result.second;
            }
            avg_similarity /= similarity_results.size();
            
            std::cout << "\n=== 统计信息 ===" << std::endl;
            std::cout << "最高相似度: " << max_similarity << std::endl;
            std::cout << "最低相似度: " << min_similarity << std::endl;
            std::cout << "平均相似度: " << avg_similarity << std::endl;
            std::cout << "图片数量: " << similarity_results.size() << std::endl;
        }
        
        // 输出性能统计
        double avg_preprocess_ms, avg_inference_ms;
        int total_count;
        rdk_inference.get_performance_stats(avg_preprocess_ms, avg_inference_ms, total_count);
        
        std::cout << "\n=== 性能统计 ===" << std::endl;
        std::cout << "平均预处理时间: " << avg_preprocess_ms << "ms" << std::endl;
        std::cout << "平均推理时间: " << avg_inference_ms << "ms" << std::endl;
        std::cout << "平均总时间: " << (avg_preprocess_ms + avg_inference_ms) << "ms" << std::endl;
        std::cout << "总处理次数: " << total_count << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}