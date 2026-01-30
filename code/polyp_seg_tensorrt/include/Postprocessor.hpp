/**
 * @file Postprocessor.hpp
 * @brief 后处理：按 Preprocessor 记录的 info 反 padding（裁掉黑边）→ 二值化 → 缩放回原图尺寸。
 */
#pragma once
#include "Preprocessor.hpp"
#include <opencv2/opencv.hpp>

class Postprocessor {
public:
    Postprocessor(int h, int w, float threshold = 0.5f);

    /** 需传入 Preprocessor 记录的 ImageInfo，先裁掉 padding 再二值化并还原到原图尺寸。 */
    cv::Mat process_with_info(const float* output_raw, const Preprocessor::ImageInfo& info);

private:
    int h_, w_;
    float threshold_;
};
