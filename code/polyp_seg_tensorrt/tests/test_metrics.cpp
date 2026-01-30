/**
 * Metrics 单元测试：Dice、IoU 公式与边界情况。
 */
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "../include/Metrics.hpp"

namespace {

TEST(Metrics, DicePerfectMatch) {
    cv::Mat pred(10, 10, CV_8UC1);
    cv::Mat gt(10, 10, CV_8UC1);
    pred.setTo(255);
    gt.setTo(255);
    float dice = Metrics::calculateDice(pred, gt);
    EXPECT_FLOAT_EQ(dice, 1.0f);
}

TEST(Metrics, DiceNoOverlap) {
    cv::Mat pred(10, 10, CV_8UC1);
    cv::Mat gt(10, 10, CV_8UC1);
    pred.setTo(0);
    gt.setTo(0);
    pred.at<uchar>(0, 0) = 255;
    gt.at<uchar>(9, 9) = 255;
    float dice = Metrics::calculateDice(pred, gt);
    EXPECT_FLOAT_EQ(dice, 0.0f);
}

TEST(Metrics, DiceHalfOverlap) {
    cv::Mat pred(2, 2, CV_8UC1);
    cv::Mat gt(2, 2, CV_8UC1);
    pred.setTo(0);
    gt.setTo(0);
    pred.at<uchar>(0, 0) = 255;
    pred.at<uchar>(0, 1) = 255;
    gt.at<uchar>(0, 0) = 255;
    gt.at<uchar>(1, 0) = 255;
    float dice = Metrics::calculateDice(pred, gt);
    // inter=1, sum_p=2, sum_g=2 -> dice = 2*1/4 = 0.5
    EXPECT_FLOAT_EQ(dice, 0.5f);
}

TEST(Metrics, DiceEmptyMasks) {
    cv::Mat pred(10, 10, CV_8UC1);
    cv::Mat gt(10, 10, CV_8UC1);
    pred.setTo(0);
    gt.setTo(0);
    float dice = Metrics::calculateDice(pred, gt);
    EXPECT_FLOAT_EQ(dice, 1.0f);
}

TEST(Metrics, IoUPerfectMatch) {
    cv::Mat pred(10, 10, CV_8UC1);
    cv::Mat gt(10, 10, CV_8UC1);
    pred.setTo(255);
    gt.setTo(255);
    float iou = Metrics::calculateIOU(pred, gt);
    EXPECT_FLOAT_EQ(iou, 1.0f);
}

TEST(Metrics, IoUNoOverlap) {
    cv::Mat pred(10, 10, CV_8UC1);
    cv::Mat gt(10, 10, CV_8UC1);
    pred.setTo(0);
    gt.setTo(0);
    pred.at<uchar>(0, 0) = 255;
    gt.at<uchar>(9, 9) = 255;
    float iou = Metrics::calculateIOU(pred, gt);
    EXPECT_FLOAT_EQ(iou, 0.0f);
}

TEST(Metrics, IoUHalfOverlap) {
    cv::Mat pred(2, 2, CV_8UC1);
    cv::Mat gt(2, 2, CV_8UC1);
    pred.setTo(0);
    gt.setTo(0);
    pred.at<uchar>(0, 0) = 255;
    pred.at<uchar>(0, 1) = 255;
    gt.at<uchar>(0, 0) = 255;
    gt.at<uchar>(1, 0) = 255;
    float iou = Metrics::calculateIOU(pred, gt);
    // inter=1, union=2+2-1=3 -> iou = 1/3
    EXPECT_NEAR(iou, 1.f / 3.f, 1e-5f);
}

TEST(Metrics, ForegroundThreshold127) {
    cv::Mat pred(1, 2, CV_8UC1);
    cv::Mat gt(1, 2, CV_8UC1);
    pred.at<uchar>(0, 0) = 127;
    pred.at<uchar>(0, 1) = 128;
    gt.setTo(255);
    float dice = Metrics::calculateDice(pred, gt);
    // 127 不算前景，128 算 -> pred 只有 1 个前景，与 gt 2 个重叠 1 -> dice = 2*1/(1+2)=2/3
    EXPECT_NEAR(dice, 2.f / 3.f, 1e-5f);
}

}  // namespace
