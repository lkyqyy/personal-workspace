/**
 * Preprocessor 单元测试：Letterbox 参数、输出尺寸、归一化范围。
 */
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../include/Preprocessor.hpp"

namespace {

TEST(Preprocessor, ImageInfoLetterbox) {
    const int H = 256, W = 256;
    Preprocessor preprocessor(H, W);
    // 原图 100x200 -> scale = 256/200 = 1.28, new_w=128, new_h=256? no: new_w=100*1.28=128, new_h=200*1.28=256
    cv::Mat img(200, 100, CV_8UC3);
    img.setTo(cv::Scalar(128, 128, 128));
    std::vector<float> buffer(3 * H * W, 0.f);
    auto info = preprocessor.process(img, buffer.data());

    EXPECT_EQ(info.raw_w, 100);
    EXPECT_EQ(info.raw_h, 200);
    float scale = static_cast<float>(H) / std::max(100, 200);
    EXPECT_EQ(info.new_w, static_cast<int>(std::round(100 * scale)));
    EXPECT_EQ(info.new_h, static_cast<int>(std::round(200 * scale)));
    EXPECT_EQ(info.x0, (W - info.new_w) / 2);
    EXPECT_EQ(info.y0, (H - info.new_h) / 2);
}

TEST(Preprocessor, OutputBufferSize) {
    const int H = 256, W = 256;
    Preprocessor preprocessor(H, W);
    cv::Mat img(64, 64, CV_8UC3);
    img.setTo(cv::Scalar(0, 0, 0));
    std::vector<float> buffer(3 * H * W, -1.f);
    preprocessor.process(img, buffer.data());
    // 归一化后值应在合理范围（ImageNet mean/std 下约 -2～2）
    float min_val = *std::min_element(buffer.begin(), buffer.end());
    float max_val = *std::max_element(buffer.begin(), buffer.end());
    EXPECT_GE(min_val, -3.f);
    EXPECT_LE(max_val, 3.f);
}

TEST(Preprocessor, SquareImageCenterPadding) {
    const int H = 256, W = 256;
    Preprocessor preprocessor(H, W);
    cv::Mat img(128, 128, CV_8UC3);
    img.setTo(cv::Scalar(255, 255, 255));
    std::vector<float> buffer(3 * H * W, 0.f);
    auto info = preprocessor.process(img, buffer.data());

    EXPECT_EQ(info.raw_w, 128);
    EXPECT_EQ(info.raw_h, 128);
    EXPECT_EQ(info.new_w, 256);
    EXPECT_EQ(info.new_h, 256);
    EXPECT_EQ(info.x0, 0);
    EXPECT_EQ(info.y0, 0);
}

}  // namespace
