/**
 * InferenceEngine 调用链：
 *   load()        → 读 .engine → Runtime/Engine/Context → 分配 d_bindings_
 *   predict(img)  → Preprocessor::process(img) → predictRaw() → Postprocessor::process_with_info()
 *   predictRaw()  → Host→Device → enqueueV2 → Device→Host（锚点校验直接调此接口）
 */
#include "InferenceEngine.hpp"
#include "Preprocessor.hpp"
#include "Postprocessor.hpp"
#include <chrono>
#include <fstream>
#include <ratio>
#include <vector>

InferenceEngine::InferenceEngine() {
    cudaStreamCreate(&stream_);
}

InferenceEngine::~InferenceEngine() {
    release();
}

void InferenceEngine::release() {
    if (stream_) cudaStreamDestroy(stream_);
    if (d_bindings_[0]) cudaFree(d_bindings_[0]);
    if (d_bindings_[1]) cudaFree(d_bindings_[1]);
    d_bindings_[0] = d_bindings_[1] = nullptr;
}

bool InferenceEngine::load(const std::string& engine_path) {
    release();
    cudaStreamCreate(&stream_);

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) return false;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size == 0) return false;  // 空文件会导致 deserialize 收到 nullptr
    std::vector<char> engine_data(size);
    if (!file.read(engine_data.data(), size)) return false;

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return false;
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) return false;  // 反序列化失败（如 GPU/TRT 版本不匹配）时避免 engine_->createExecutionContext() 段错误
    context_.reset(engine_->createExecutionContext());

    if (!context_) return false;

    auto input_dims = engine_->getTensorShape(engine_->getBindingName(0));
    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];

    input_size_bytes_ = 1 * 3 * input_h_ * input_w_ * sizeof(float);
    output_size_bytes_ = 1 * 1 * input_h_ * input_w_ * sizeof(float);

    cudaMalloc(&d_bindings_[0], input_size_bytes_);
    cudaMalloc(&d_bindings_[1], output_size_bytes_);

    return true;
}

bool InferenceEngine::predictRaw(const float* input, float* output) {
    if (!isLoaded()) return false;

    cudaMemcpyAsync(d_bindings_[0], input, input_size_bytes_, cudaMemcpyHostToDevice, stream_);

    context_->setInputShape(engine_->getBindingName(0), nvinfer1::Dims4{1, 3, input_h_, input_w_});
    context_->enqueueV2(d_bindings_, stream_, nullptr);

    cudaMemcpyAsync(output, d_bindings_[1], output_size_bytes_, cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
    return true;
}

// 异常输入防护：空图、非法尺寸、超大图（避免 letterbox 除零或 OOM）
static bool isValidInput(const cv::Mat& img) {
    if (img.empty() || img.rows <= 0 || img.cols <= 0) return false;
    const int kMaxDim = 8192;
    if (img.rows > kMaxDim || img.cols > kMaxDim) return false;
    return true;
}

bool InferenceEngine::predict(const cv::Mat& img, cv::Mat& mask_out) {
    if (!isValidInput(img)) return false;
    Preprocessor preprocessor(input_h_, input_w_);
    std::vector<float> input_tensor(static_cast<size_t>(3) * input_h_ * input_w_);
    Preprocessor::ImageInfo info = preprocessor.process(img, input_tensor.data());
    std::vector<float> prob_out(static_cast<size_t>(input_h_) * input_w_);
    if (!predictRaw(input_tensor.data(), prob_out.data())) return false;
    Postprocessor postprocessor(input_h_, input_w_, threshold_);
    mask_out = postprocessor.process_with_info(prob_out.data(), info);
    return true;
}

bool InferenceEngine::predictWithTiming(const cv::Mat& img, cv::Mat& mask_out, TimingResult& timing) {
    if (!isValidInput(img)) return false;
    timing = TimingResult{};
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    Preprocessor preprocessor(input_h_, input_w_);
    std::vector<float> input_tensor(static_cast<size_t>(3) * input_h_ * input_w_);
    Preprocessor::ImageInfo info = preprocessor.process(img, input_tensor.data());
    auto t1 = high_resolution_clock::now();
    std::vector<float> prob_out(static_cast<size_t>(input_h_) * input_w_);
    if (!predictRaw(input_tensor.data(), prob_out.data())) return false;
    auto t2 = high_resolution_clock::now();
    Postprocessor postprocessor(input_h_, input_w_, threshold_);
    mask_out = postprocessor.process_with_info(prob_out.data(), info);
    auto t3 = high_resolution_clock::now();
    timing.pre_ms = duration<float, std::milli>(t1 - t0).count();
    timing.infer_ms = duration<float, std::milli>(t2 - t1).count();
    timing.post_ms = duration<float, std::milli>(t3 - t2).count();
    return true;
}
