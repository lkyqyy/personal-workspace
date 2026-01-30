/**
 * Postprocessor 单元测试：裁边、二值化阈值、输出尺寸。
 */
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../include/Postprocessor.hpp"
#include "../include/Preprocessor.hpp"

namespace {

TEST(Postprocessor, OutputShapeMatchesRawSize) {
    const int H = 256, W = 256;
    Postprocessor postprocessor(H, W, 0.5f);
    Preprocessor::ImageInfo info;
    info.raw_w = 100;
    info.raw_h = 200;
    info.new_w = 128;
    info.new_h = 256;
    info.x0 = (W - 128) / 2;
    info.y0 = 0;

    std::vector<float> prob(H * W, 0.f);
    for (int y = info.y0; y < info.y0 + info.new_h; ++y)
        for (int x = info.x0; x < info.x0 + info.new_w; ++x)
            prob[y * W + x] = 0.6f;
    cv::Mat mask = postprocessor.process_with_info(prob.data(), info);
    EXPECT_EQ(mask.cols, info.raw_w);
    EXPECT_EQ(mask.rows, info.raw_h);
    EXPECT_EQ(mask.type(), CV_8UC1);
}

TEST(Postprocessor, ThresholdBinary) {
    const int H = 4, W = 4;
    Postprocessor postprocessor(H, W, 0.5f);
    Preprocessor::ImageInfo info;
    info.raw_w = 4;
    info.raw_h = 4;
    info.new_w = 4;
    info.new_h = 4;
    info.x0 = 0;
    info.y0 = 0;

    std::vector<float> prob(H * W);
    for (int i = 0; i < H * W; ++i)
        prob[i] = (i % 2 == 0) ? 0.6f : 0.4f;
    cv::Mat mask = postprocessor.process_with_info(prob.data(), info);
    EXPECT_EQ(mask.cols, 4);
    EXPECT_EQ(mask.rows, 4);
    int above = 0;
    for (int i = 0; i < mask.total(); ++i)
        if (mask.at<uchar>(i) > 127) above++;
    EXPECT_EQ(above, 8);
}

}  // namespace
