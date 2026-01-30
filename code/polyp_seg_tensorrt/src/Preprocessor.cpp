#include "Preprocessor.hpp"
#include <algorithm>
#include <cmath>

Preprocessor::Preprocessor(int target_h, int target_w) : h_(target_h), w_(target_w) {}

/** PIL BILINEAR 风格：首末像素对齐，src = i * (src_size-1)/(dst_size-1)，再双线性插值。 */
static void resizePilBilinear(const cv::Mat& src, cv::Mat& dst, int dst_w, int dst_h) {
    dst.create(dst_h, dst_w, src.type());
    const int src_w = src.cols, src_h = src.rows;
    const int channels = src.channels();
    if (dst_w <= 0 || dst_h <= 0) return;
    const float fx = (dst_w > 1) ? static_cast<float>(src_w - 1) / (dst_w - 1) : 0.f;
    const float fy = (dst_h > 1) ? static_cast<float>(src_h - 1) / (dst_h - 1) : 0.f;
    for (int y = 0; y < dst_h; ++y) {
        float src_y = (dst_h > 1) ? y * fy : static_cast<float>(src_h - 1) * 0.5f;
        int iy0 = std::max(0, std::min(static_cast<int>(src_y), src_h - 1));
        int iy1 = std::max(0, std::min(iy0 + 1, src_h - 1));
        float wy = src_y - iy0;
        for (int x = 0; x < dst_w; ++x) {
            float src_x = (dst_w > 1) ? x * fx : static_cast<float>(src_w - 1) * 0.5f;
            int ix0 = std::max(0, std::min(static_cast<int>(src_x), src_w - 1));
            int ix1 = std::max(0, std::min(ix0 + 1, src_w - 1));
            float wx = src_x - ix0;
            for (int c = 0; c < channels; ++c) {
                float v00 = src.at<cv::Vec3b>(iy0, ix0)[c];
                float v10 = src.at<cv::Vec3b>(iy0, ix1)[c];
                float v01 = src.at<cv::Vec3b>(iy1, ix0)[c];
                float v11 = src.at<cv::Vec3b>(iy1, ix1)[c];
                float v = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10
                       + (1 - wx) * wy * v01 + wx * wy * v11;
                dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(std::round(std::max(0.f, std::min(255.f, v))));
            }
        }
    }
}

Preprocessor::ImageInfo Preprocessor::process(const cv::Mat& img, float* buffer) {
    ImageInfo info;
    info.raw_w = img.cols;
    info.raw_h = img.rows;

    float scale = static_cast<float>(h_) / std::max(info.raw_w, info.raw_h);
    info.new_w = static_cast<int>(std::round(info.raw_w * scale));
    info.new_h = static_cast<int>(std::round(info.raw_h * scale));

    cv::Mat resized;
    resizePilBilinear(img, resized, info.new_w, info.new_h);

    cv::Mat canvas = cv::Mat::zeros(cv::Size(w_, h_), CV_8UC3);
    info.x0 = (w_ - info.new_w) / 2;
    info.y0 = (h_ - info.new_h) / 2;
    resized.copyTo(canvas(cv::Rect(info.x0, info.y0, info.new_w, info.new_h)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

    int area = h_ * w_;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < area; ++i) {
            float val = static_cast<float>(rgb.data[i * 3 + c]) / 255.0f;
            buffer[c * area + i] = (val - mean_[c]) / std_[c];
        }
    }
    return info;
}
