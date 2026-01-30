#include "Postprocessor.hpp"

Postprocessor::Postprocessor(int h, int w, float threshold)
    : h_(h), w_(w), threshold_(threshold) {}

cv::Mat Postprocessor::process_with_info(const float* output_raw, const Preprocessor::ImageInfo& info) {
    cv::Mat prob_mat(h_, w_, CV_32FC1, const_cast<float*>(output_raw));

    cv::Mat cropped_prob = prob_mat(cv::Rect(info.x0, info.y0, info.new_w, info.new_h));

    cv::Mat mask;
    cv::threshold(cropped_prob, mask, threshold_, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1);

    cv::Mat final_mask;
    cv::resize(mask, final_mask, cv::Size(info.raw_w, info.raw_h), 0, 0, cv::INTER_NEAREST);

    return final_mask;
}
