#pragma once
#include "common.hpp"
#include <opencv2/opencv.hpp>

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // 禁止拷贝，防止显存多次释放
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    /** 加载模型并预分配 GPU 资源 */
    bool load(const std::string& engine_path);

    /** 生产接口：端到端推理 */
    bool predict(const cv::Mat& img, cv::Mat& mask_out);

    /** 性能分析：与 predict 相同，并回填各阶段耗时（毫秒）。 */
    struct TimingResult {
        float pre_ms = 0.f;   // 预处理（letterbox + 归一化）
        float infer_ms = 0.f; // 推理（H2D + kernel + D2H）
        float post_ms = 0.f;   // 后处理（裁边 + 二值化 + resize）
    };
    bool predictWithTiming(const cv::Mat& img, cv::Mat& mask_out, TimingResult& timing);

    /** 锚点校验接口：裸数据对比 */
    bool predictRaw(const float* input, float* output);

    bool isLoaded() const { return context_ != nullptr; }

private:
    // TensorRT 核心组件
    Logger logger_;
    TRTPtr<nvinfer1::IRuntime> runtime_ = nullptr;
    TRTPtr<nvinfer1::ICudaEngine> engine_ = nullptr;
    TRTPtr<nvinfer1::IExecutionContext> context_ = nullptr;

    // CUDA 资源 (RAII 管理)
    cudaStream_t stream_ = nullptr;
    void* d_bindings_[2] = {nullptr, nullptr}; // 0: Input, 1: Output

    // 模型元数据
    int input_h_ = 256;
    int input_w_ = 256;
    size_t input_size_bytes_ = 0;
    size_t output_size_bytes_ = 0;

    // 图像预处理参数
    const float mean_[3] = {0.485f, 0.456f, 0.406f};
    const float std_[3]  = {0.229f, 0.224f, 0.225f};
    const float threshold_ = 0.5f;

    void release();
};