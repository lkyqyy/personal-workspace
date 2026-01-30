/**
 * 锚点校验 + 端到端验证：先 predictRaw 与 output_prob.bin 逐元素对比，再 predict 算 Dice/IoU。
 * 单图调试：传入 --first-5 时从 data/images 自动取前 5 张并保存到 debug_cpp/img_0..img_4；
 *           或传入 1～N 个图片路径，保存到 debug_cpp/ 或 debug_cpp/img_0..img_{N-1}。
 * 锚点 bin 格式与 scripts/generate_anchor_data.py 一致：前 4 个 int32 为 shape (N,C,H,W)，随后为 float32 数据。
 */
#include "InferenceEngine.hpp"
#include "Metrics.hpp"
#include "Preprocessor.hpp"
#include "Postprocessor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/** 写出与 Python debug_single_image 一致的 bin：4×int32 shape + NCHW float32。 */
static void save_bin_with_shape(const std::string& path, const float* data, int n, int c, int h, int w) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return;
    int32_t shape[4] = {n, c, h, w};
    ofs.write(reinterpret_cast<const char*>(shape), 16);
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * c * h * w) * sizeof(float));
}

/** 加载锚点 bin（格式：4×int32 shape + NCHW float32），返回 float 数据；shape 可选回填。 */
static std::vector<float> load_anchor_bin(const std::string& path, int* out_n = nullptr,
                                          int* out_c = nullptr, int* out_h = nullptr, int* out_w = nullptr) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return {};
    const size_t file_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    if (file_size < 16u) return {};
    int32_t shape[4];
    ifs.read(reinterpret_cast<char*>(shape), 16);
    size_t num_floats = 1;
    for (int i = 0; i < 4; ++i) num_floats *= static_cast<size_t>(shape[i]);
    if (file_size < 16u + num_floats * sizeof(float)) return {};
    std::vector<float> data(num_floats);
    ifs.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));
    if (out_n) *out_n = shape[0];
    if (out_c) *out_c = shape[1];
    if (out_h) *out_h = shape[2];
    if (out_w) *out_w = shape[3];
    return data;
}

int main(int argc, char** argv) {
    const std::string engine_path = "models/unet_polyp_seg_fp16.engine";  // 用 fp32 时改为 unet_polyp_seg_fp32.engine
    const std::string input_bin  = "models/anchor/input_tensor.bin";
    const std::string output_bin = "models/anchor/output_prob.bin";
    const std::string test_image = "data/test_polyp.jpg";
    const std::string test_mask  = "data/test_polyp_gt.png";

    InferenceEngine engine;
    if (!engine.load(engine_path)) {
        std::cerr << "Engine load failed: " << engine_path << std::endl;
        std::cerr << "  (run from project root; engine must exist and match this GPU/TRT version)" << std::endl;
        return -1;
    }
    std::cout << "Loaded engine: " << engine_path << std::endl;  // 用于确认当前为 fp16 或 fp32

    // 性能瓶颈分析：--benchmark [轮数]，在完整验证集上跑指定轮数，汇总各阶段耗时，默认 20 轮
    if (argc >= 2 && std::string(argv[1]) == "--benchmark") {
        int rounds = 20;
        if (argc >= 3) { rounds = std::atoi(argv[2]); if (rounds < 1) rounds = 20; }
        const std::string val_images_dir = "data/images";
        std::vector<std::string> image_paths;
        cv::glob(val_images_dir + "/*.jpg", image_paths);
        {
            std::vector<std::string> png_paths;
            cv::glob(val_images_dir + "/*.png", png_paths);
            image_paths.insert(image_paths.end(), png_paths.begin(), png_paths.end());
        }
        std::sort(image_paths.begin(), image_paths.end());
        if (image_paths.empty()) {
            std::cerr << "No images in " << val_images_dir << " for benchmark." << std::endl;
            return -1;
        }
        const size_t num_images = image_paths.size();
        const int warmup = std::min(10, static_cast<int>(num_images));
        std::cout << "[Benchmark] " << num_images << " images, warmup " << warmup << "..." << std::endl;
        for (int i = 0; i < warmup; ++i) {
            cv::Mat img = cv::imread(image_paths[static_cast<size_t>(i)]);
            if (!img.empty()) { cv::Mat mask; engine.predict(img, mask); }
        }
        std::cout << "  warmup done. " << rounds << " rounds..." << std::endl;
        std::vector<float> pre_ms, infer_ms, post_ms;
        pre_ms.reserve(num_images * static_cast<size_t>(rounds));
        infer_ms.reserve(num_images * static_cast<size_t>(rounds));
        post_ms.reserve(num_images * static_cast<size_t>(rounds));
        for (int r = 0; r < rounds; ++r) {
            if (r % 5 == 0 || r == rounds - 1)
                std::cout << "  round " << (r + 1) << "/" << rounds << " ..." << std::endl;
            for (const std::string& img_path : image_paths) {
                cv::Mat img = cv::imread(img_path);
                if (img.empty()) continue;
                cv::Mat mask;
                InferenceEngine::TimingResult tr;
                if (!engine.predictWithTiming(img, mask, tr)) continue;
                pre_ms.push_back(tr.pre_ms);
                infer_ms.push_back(tr.infer_ms);
                post_ms.push_back(tr.post_ms);
            }
        }
        if (pre_ms.empty()) {
            std::cerr << "No valid inference on validation set." << std::endl;
            return -1;
        }
        const int n = static_cast<int>(pre_ms.size());
        auto mean_std = [](const std::vector<float>& v, float& mean, float& stddev) {
            double sum = 0, sum2 = 0;
            for (float x : v) { sum += x; sum2 += x * x; }
            mean = static_cast<float>(sum / static_cast<int>(v.size()));
            stddev = static_cast<float>(std::sqrt(sum2 / static_cast<int>(v.size()) - mean * mean));
            if (stddev < 0.f) stddev = 0.f;
        };
        float m_pre, s_pre, m_infer, s_infer, m_post, s_post;
        mean_std(pre_ms, m_pre, s_pre);
        mean_std(infer_ms, m_infer, s_infer);
        mean_std(post_ms, m_post, s_post);
        float m_total = m_pre + m_infer + m_post;
        std::cout << "\n[Benchmark] validation set, " << static_cast<int>(num_images) << " images x " << rounds << " rounds (warmup " << warmup << "), " << n << " samples" << std::endl;
        std::cout << "  Stage       | Mean (ms) | Std (ms) | % of total" << std::endl;
        std::cout << "  ------------|-----------|----------|------------" << std::endl;
        std::cout << "  Preprocess  | " << std::fixed << std::setprecision(3) << std::setw(8) << m_pre << " | " << std::setw(7) << s_pre << " | " << std::setprecision(1) << (m_total > 0 ? (m_pre / m_total * 100.f) : 0.f) << "%" << std::endl;
        std::cout << "  Inference   | " << std::setprecision(3) << std::setw(8) << m_infer << " | " << std::setw(7) << s_infer << " | " << std::setprecision(1) << (m_total > 0 ? (m_infer / m_total * 100.f) : 0.f) << "%" << std::endl;
        std::cout << "  Postprocess  | " << std::setprecision(3) << std::setw(8) << m_post << " | " << std::setw(7) << s_post << " | " << std::setprecision(1) << (m_total > 0 ? (m_post / m_total * 100.f) : 0.f) << "%" << std::endl;
        std::cout << "  ------------|-----------|----------|------------" << std::endl;
        std::cout << "  Total       | " << std::setprecision(3) << std::setw(8) << m_total << " | (end-to-end)" << std::endl;
        const char* bottleneck = (m_infer >= m_pre && m_infer >= m_post) ? "Inference (GPU)" : (m_pre >= m_post ? "Preprocess (CPU)" : "Postprocess (CPU)");
        std::cout << "  Bottleneck  | " << bottleneck << std::endl;
        std::cout << std::endl;
        return 0;
    }

    // 压力测试：视频流模拟 + 吞吐/延迟/P99 + 按分钟 FPS（热稳定性观察）
    if (argc >= 2 && std::string(argv[1]) == "--stress") {
        int duration_sec = 600;  // 默认 10 分钟
        if (argc >= 3) { duration_sec = std::atoi(argv[2]); if (duration_sec < 1) duration_sec = 600; }
        const std::string val_images_dir = "data/images";
        std::vector<std::string> image_paths;
        cv::glob(val_images_dir + "/*.jpg", image_paths);
        {
            std::vector<std::string> png_paths;
            cv::glob(val_images_dir + "/*.png", png_paths);
            image_paths.insert(image_paths.end(), png_paths.begin(), png_paths.end());
        }
        std::sort(image_paths.begin(), image_paths.end());
        if (image_paths.empty()) {
            std::cerr << "No images in " << val_images_dir << " for stress test." << std::endl;
            return -1;
        }
        std::vector<cv::Mat> images;
        images.reserve(image_paths.size());
        for (const std::string& p : image_paths) {
            cv::Mat img = cv::imread(p);
            if (!img.empty()) images.push_back(img);
        }
        if (images.empty()) {
            std::cerr << "No valid images loaded for stress test." << std::endl;
            return -1;
        }
        std::cout << "\n[Stress] " << static_cast<int>(images.size()) << " images, " << duration_sec << " s (throughput + latency P99 + FPS/min)" << std::endl;
        std::cout << "  Tip: run \"watch -n 1 nvidia-smi\" in another terminal to monitor VRAM." << std::endl;
        using Clock = std::chrono::steady_clock;
        const auto t_start = Clock::now();
        int64_t frame_count = 0;
        int error_count = 0;
        std::vector<float> latency_ms;
        const int latency_sample_interval = 50;  // 每 50 帧采样一次延迟
        latency_ms.reserve(static_cast<size_t>(duration_sec * 300));  // 约 300 FPS 上限
        auto t_last_min = t_start;
        int64_t frames_last_min = 0;
        while (true) {
            auto t_now = Clock::now();
            const int64_t elapsed = std::chrono::duration_cast<std::chrono::seconds>(t_now - t_start).count();
            if (elapsed >= duration_sec) break;
            const cv::Mat& img = images[static_cast<size_t>(frame_count % images.size())];
            cv::Mat mask;
            if (frame_count % latency_sample_interval == 0) {
                InferenceEngine::TimingResult tr;
                if (engine.predictWithTiming(img, mask, tr)) {
                    float total = tr.pre_ms + tr.infer_ms + tr.post_ms;
                    latency_ms.push_back(total);
                } else
                    ++error_count;
            } else {
                if (!engine.predict(img, mask)) ++error_count;
            }
            ++frame_count;
            frames_last_min++;
            const double sec_since_last = std::chrono::duration<double>(t_now - t_last_min).count();
            if (sec_since_last >= 60.0) {
                const double fps_min = (sec_since_last > 0) ? (static_cast<double>(frames_last_min) / sec_since_last) : 0.0;
                std::cout << "  [" << (elapsed / 60) << " min] FPS (last 60s): " << std::fixed << std::setprecision(2) << fps_min << std::endl;
                t_last_min = t_now;
                frames_last_min = 0;
            }
        }
        const auto t_end = Clock::now();
        const double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
        const double fps = (elapsed_sec > 0 && frame_count > 0) ? (static_cast<double>(frame_count) / elapsed_sec) : 0.0;
        std::cout << "  ---" << std::endl;
        std::cout << "  Frames: " << frame_count << "  Elapsed: " << std::fixed << std::setprecision(2) << elapsed_sec << " s  FPS: " << std::setprecision(2) << fps << std::endl;
        std::cout << "  Errors: " << error_count << std::endl;
        if (!latency_ms.empty()) {
            std::sort(latency_ms.begin(), latency_ms.end());
            const int n = static_cast<int>(latency_ms.size());
            double sum = 0;
            for (float x : latency_ms) sum += x;
            float mean = static_cast<float>(sum / n);
            int p99_idx = static_cast<int>(std::ceil(0.99 * n)) - 1;
            if (p99_idx < 0) p99_idx = 0;
            float p99 = latency_ms[static_cast<size_t>(p99_idx)];
            float max_lat = latency_ms.back();
            std::cout << "  Latency: mean " << std::fixed << std::setprecision(2) << mean << " ms  P99 " << p99 << " ms  max " << max_lat << " ms  (n=" << n << ")" << std::endl;
        }
        if (error_count > 0) std::cout << "  \033[31mWARNING: " << error_count << " inference failures\033[0m" << std::endl;
        std::cout << std::endl;
        return 0;
    }

    // 异常数据抗压：空图、全黑/全白、极端长宽比、非法尺寸，期望不崩溃且对非法输入返回 false
    if (argc >= 2 && std::string(argv[1]) == "--stress-robustness") {
        std::cout << "\n[Stress-Robustness] garbage input tests (expect no crash; invalid input -> false)" << std::endl;
        cv::Mat mask;
        int passed = 0, failed = 0;
        auto test_one = [&](const std::string& name, const cv::Mat& img, bool expect_ok) {
            cv::Mat out;
            bool ok = engine.predict(img, out);
            if ((ok == expect_ok)) {
                std::cout << "  \033[32mPASS\033[0m " << name << std::endl;
                ++passed;
            } else {
                std::cout << "  \033[31mFAIL\033[0m " << name << " (got " << (ok ? "true" : "false") << ", expect " << (expect_ok ? "true" : "false") << ")" << std::endl;
                ++failed;
            }
        };
        cv::Mat empty_mat;
        test_one("empty Mat", empty_mat, false);
        cv::Mat black_256(256, 256, CV_8UC3, cv::Scalar(0, 0, 0));
        test_one("all black 256x256", black_256, true);
        cv::Mat white_256(256, 256, CV_8UC3, cv::Scalar(255, 255, 255));
        test_one("all white 256x256", white_256, true);
        cv::Mat extreme(10, 10000, CV_8UC3);  // 极端长宽比，超过 8192 会被拒绝
        extreme.setTo(cv::Scalar(128, 128, 128));
        test_one("extreme 10000x10 (reject >8192)", extreme, false);
        cv::Mat zero_rows(0, 256, CV_8UC3);
        test_one("0 rows", zero_rows, false);
        cv::Mat zero_cols(256, 0, CV_8UC3);
        test_one("0 cols", zero_cols, false);
        std::cout << "  --- " << passed << " passed, " << failed << " failed" << std::endl;
        std::cout << std::endl;
        return failed > 0 ? -1 : 0;
    }

    // 单图调试：--first-5 时从 data/images 自动取前 5 张（与 Python sorted 一致）；或传入 1～N 个图片路径
    if (argc >= 2) {
        std::vector<std::string> image_paths;
        if (argc == 2 && std::string(argv[1]) == "--first-5") {
            const std::string val_dir = "data/images";
            std::vector<std::string> jpg, png;
            cv::glob(val_dir + "/*.jpg", jpg);
            cv::glob(val_dir + "/*.png", png);
            image_paths.insert(image_paths.end(), jpg.begin(), jpg.end());
            image_paths.insert(image_paths.end(), png.begin(), png.end());
            std::sort(image_paths.begin(), image_paths.end());
            if (image_paths.size() > 5) image_paths.resize(5);
            if (image_paths.empty()) {
                std::cerr << "No images in " << val_dir << std::endl;
                return -1;
            }
            std::cout << "[C++] --first-5: using " << static_cast<int>(image_paths.size()) << " images from " << val_dir << std::endl;
        } else {
            for (int i = 1; i < argc; ++i) image_paths.push_back(argv[i]);
        }
        (void)std::system("mkdir -p debug_cpp");
        const int H = 256, W = 256;
        Preprocessor preprocessor(H, W);
        Postprocessor postprocessor(H, W, 0.5f);
        std::vector<float> input_tensor(static_cast<size_t>(3) * H * W);
        std::vector<float> prob_out(static_cast<size_t>(H) * W);
        const int num_images = static_cast<int>(image_paths.size());
        for (int idx = 0; idx < num_images; ++idx) {
            const std::string& img_path = image_paths[static_cast<size_t>(idx)];
            cv::Mat img = cv::imread(img_path);
            if (img.empty()) {
                std::cerr << "Failed to read image: " << img_path << std::endl;
                return -1;
            }
            std::string out_dir = (num_images > 1) ? ("debug_cpp/img_" + std::to_string(idx)) : "debug_cpp";
            (void)std::system(("mkdir -p " + out_dir).c_str());
            Preprocessor::ImageInfo info = preprocessor.process(img, input_tensor.data());
            save_bin_with_shape(out_dir + "/input.bin", input_tensor.data(), 1, 3, H, W);
            if (!engine.predictRaw(input_tensor.data(), prob_out.data())) return -1;
            save_bin_with_shape(out_dir + "/prob.bin", prob_out.data(), 1, 1, H, W);
            cv::Mat mask = postprocessor.process_with_info(prob_out.data(), info);
            cv::imwrite(out_dir + "/mask.png", mask);
            std::ofstream f_letterbox(out_dir + "/letterbox.txt");
            if (f_letterbox)
                f_letterbox << info.raw_w << " " << info.raw_h << " " << info.new_w << " " << info.new_h << " " << info.x0 << " " << info.y0 << "\n";
            std::ofstream f_stats(out_dir + "/input_stats.txt");
            if (f_stats) {
                for (int c = 0; c < 3; ++c) {
                    double sum = 0, minv = 1e30, maxv = -1e30;
                    for (int i = 0; i < H * W; ++i) {
                        float v = input_tensor[static_cast<size_t>(c) * H * W + i];
                        sum += v;
                        if (v < minv) minv = v;
                        if (v > maxv) maxv = v;
                    }
                    f_stats << "ch" << c << " min=" << std::fixed << std::setprecision(6) << minv << " max=" << maxv << " mean=" << (sum / (H * W)) << "\n";
                }
            }
            std::cout << "[C++] " << img_path << " -> " << out_dir << "/" << std::endl;
        }
        return 0;
    }

    std::cout << "\n[Step 1] Anchor Validation (Raw Tensor Match)..." << std::endl;
    int anchor_n = 0, anchor_c = 0, anchor_h = 0, anchor_w = 0;
    auto h_input_anchor  = load_anchor_bin(input_bin, &anchor_n, &anchor_c, &anchor_h, &anchor_w);
    auto h_golden_anchor = load_anchor_bin(output_bin);
    if (!h_input_anchor.empty() && !h_golden_anchor.empty()) {
        if (anchor_n != 1) {
            std::cout << " -> Skip: predictRaw 仅支持 batch=1，当前 anchor N=" << anchor_n
                      << "；请运行 generate_anchor_data.py --num 1 重新生成后再校验" << std::endl;
        } else {
            std::vector<float> h_trt_raw(h_golden_anchor.size());
            engine.predictRaw(h_input_anchor.data(), h_trt_raw.data());

            double mae = 0;
            for (size_t i = 0; i < h_trt_raw.size(); ++i) {
                mae += std::abs(h_trt_raw[i] - static_cast<double>(h_golden_anchor[i]));
            }
            mae /= static_cast<double>(h_trt_raw.size());
            std::cout << " -> Anchor MAE: " << std::fixed << std::setprecision(10) << mae << std::endl;
            // TRT vs PyTorch/ONNX 常有 ~1e-3 量级差异，用 1e-2 作为可接受阈值
            if (mae < 1e-2) std::cout << " -> \033[32mPASS\033[0m: Within acceptable precision." << std::endl;
            else std::cout << " -> \033[31mFAIL\033[0m: Precision gap detected!" << std::endl;
        }
    } else {
        std::cout << " -> Skip (missing " << input_bin << " or " << output_bin << ")" << std::endl;
    }

    std::cout << "\n[Step 2] End-to-End Validation (Image -> Mask)..." << std::endl;

    const std::string val_images_dir = "data/images";
    const std::string val_masks_dir  = "data/masks";
    std::vector<std::string> image_paths;
    cv::glob(val_images_dir + "/*.jpg", image_paths);
    {
        std::vector<std::string> png_paths;
        cv::glob(val_images_dir + "/*.png", png_paths);
        image_paths.insert(image_paths.end(), png_paths.begin(), png_paths.end());
    }
    std::sort(image_paths.begin(), image_paths.end());  // 与 Python sorted(glob) 一致，便于逐张对比

    if (!image_paths.empty()) {
        float total_dice = 0.f, total_iou = 0.f;
        int count = 0;
        for (const std::string& img_path : image_paths) {
            std::string stem = img_path;
            size_t slash = stem.find_last_of("/\\");
            if (slash != std::string::npos) stem = stem.substr(slash + 1);
            size_t dot = stem.find_last_of('.');
            if (dot != std::string::npos) stem = stem.substr(0, dot);
            std::string mask_path = val_masks_dir + "/" + stem + ".png";

            cv::Mat img = cv::imread(img_path);
            cv::Mat gt  = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
            if (img.empty() || gt.empty()) continue;

            cv::Mat pred_mask;
            if (!engine.predict(img, pred_mask)) continue;

            total_dice += Metrics::calculateDice(pred_mask, gt);
            total_iou  += Metrics::calculateIOU(pred_mask, gt);
            ++count;
        }
        if (count > 0) {
            std::cout << " -> Validation set: " << count << " images" << std::endl;
            std::cout << " -> Mean Dice: " << std::fixed << std::setprecision(4) << (total_dice / count) << std::endl;
            std::cout << " -> Mean IoU:  " << (total_iou / count) << std::endl;
        } else {
            std::cout << " -> No valid (img+gt) pairs in " << val_images_dir << std::endl;
        }
    } else {
        cv::Mat img = cv::imread(test_image);
        cv::Mat gt  = cv::imread(test_mask, cv::IMREAD_GRAYSCALE);
        if (!img.empty() && !gt.empty()) {
            cv::Mat pred_mask;
            engine.predict(img, pred_mask);
            float dice = Metrics::calculateDice(pred_mask, gt);
            float iou  = Metrics::calculateIOU(pred_mask, gt);
            std::cout << " -> Single image Dice: " << std::setprecision(4) << dice << "  IoU: " << iou << std::endl;
            cv::imwrite("result_cpp.png", pred_mask);
            std::cout << " -> Saved result_cpp.png" << std::endl;
        } else {
            std::cout << " -> Skip (no data/images/*.jpg and missing " << test_image << " / " << test_mask << ")" << std::endl;
        }
    }

    return 0;
}
