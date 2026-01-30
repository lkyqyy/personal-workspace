#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <vector>

// TensorRT 智能指针定制销毁器
struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) obj->destroy();
    }
};

template <typename T>
using TRTPtr = std::unique_ptr<T, TRTDeleter>;

// 严格的日志记录器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR)
            std::cerr << "\033[31m[ERROR]\033[0m " << msg << std::endl;
        else if (severity <= Severity::kWARNING)
            std::cout << "\033[33m[WARN]\033[0m " << msg << std::endl;
    }
};