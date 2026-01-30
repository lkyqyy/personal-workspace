/**
 * @file Preprocessor.hpp
 * @brief Letterbox 预处理：等比缩放 + padding，记录缩放比例与偏移供后处理裁掉 padding 区域。
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class Preprocessor {
public:
    Preprocessor(int target_h, int target_w);

    /** process 返回偏移信息，供后处理还原坐标 / 裁掉 padding。 */
    struct ImageInfo {
        int raw_w;
        int raw_h;
        int new_w;
        int new_h;
        int x0;
        int y0;
    };

    ImageInfo process(const cv::Mat& img, float* buffer);

private:
    int h_, w_;
    const float mean_[3] = {0.485f, 0.456f, 0.406f};
    const float std_[3]  = {0.229f, 0.224f, 0.225f};
};
