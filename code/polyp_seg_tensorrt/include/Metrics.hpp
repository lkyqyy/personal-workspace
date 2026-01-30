/**
 * @file Metrics.hpp
 * @brief 精度矫正与测量：Dice、IoU 计算逻辑，与训练侧公式一致，用于验证集/锚点校验。
 */
#pragma once
#include <opencv2/opencv.hpp>

class Metrics {
public:
    /** 计算单张图的 Dice 系数；mask 单通道，CV_8UC1(0/255) 或 CV_32F(0~1)，与训练公式一致。 */
    static float calculateDice(const cv::Mat& maskPred, const cv::Mat& maskGT);

    /** 计算单张图的 IoU。 */
    static float calculateIOU(const cv::Mat& maskPred, const cv::Mat& maskGT);

    /** 平均指标（用于全量验证集测试）。 */
    struct Result {
        float meanDice = 0.0f;
        float meanIOU  = 0.0f;
    };
};
