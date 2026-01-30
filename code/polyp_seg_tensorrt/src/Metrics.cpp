#include "Metrics.hpp"

float Metrics::calculateDice(const cv::Mat& maskPred, const cv::Mat& maskGT) {
    cv::Mat p = (maskPred > 127);
    cv::Mat g = (maskGT > 127);
    float intersection = static_cast<float>(cv::countNonZero(p & g));
    float sum = static_cast<float>(cv::countNonZero(p) + cv::countNonZero(g));
    if (sum == 0) return 1.0f;  // 空对空约定为 1，避免除 0
    return (2.0f * intersection) / sum;
}

float Metrics::calculateIOU(const cv::Mat& maskPred, const cv::Mat& maskGT) {
    cv::Mat p = (maskPred > 127);
    cv::Mat g = (maskGT > 127);
    float intersection = static_cast<float>(cv::countNonZero(p & g));
    float union_ = static_cast<float>(cv::countNonZero(p) + cv::countNonZero(g) - cv::countNonZero(p & g));
    if (union_ == 0) return 1.0f;  // 空对空约定为 1，避免除 0
    return intersection / union_;
}
