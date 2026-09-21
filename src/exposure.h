#pragma once
#include <torch/torch.h>
#include <map>
#include <memory>
#include <fstream>
#include <iomanip>

// Scalar log-gain and bias per training keyframe. The first frame fixes the gauge.
class ExposureCompensation {
    struct Frame {
        torch::Tensor affine;
        std::unique_ptr<torch::optim::Adam> optimizer;
    };
    std::map<double, Frame> frames_;
    double learning_rate_;
public:
    explicit ExposureCompensation(double learning_rate) : learning_rate_(learning_rate) {}
    void add(double timestamp) {
        if (frames_.count(timestamp)) return;
        const bool trainable = !frames_.empty();
        Frame frame;
        frame.affine = torch::zeros({2}, torch::TensorOptions().dtype(torch::kFloat32)
            .device(torch::kCUDA).requires_grad(trainable));
        if (trainable)
            frame.optimizer = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{frame.affine}, torch::optim::AdamOptions(learning_rate_));
        frames_.emplace(timestamp, std::move(frame));
    }
    torch::Tensor parameters(double timestamp) const {
        if (frames_.empty()) return {};
        auto next = frames_.lower_bound(timestamp);
        if (next == frames_.end()) return frames_.rbegin()->second.affine;
        if (next == frames_.begin() || next->first == timestamp) return next->second.affine;
        auto prev = std::prev(next);
        double fraction = (timestamp - prev->first) / (next->first - prev->first);
        return (1. - fraction) * prev->second.affine + fraction * next->second.affine;
    }
    torch::Tensor correct(const torch::Tensor& image, double timestamp) const {
        auto affine = parameters(timestamp);
        if (!affine.defined()) return image;
        return affine[0].exp() * image + affine[1];
    }
    void step(double timestamp) {
        auto& optimizer = frames_.at(timestamp).optimizer;
        if (optimizer) { optimizer->step(); optimizer->zero_grad(true); }
    }
    void save(const std::string& path) const {
        std::ofstream csv(path);
        csv << "timestamp,log_gain,bias\n" << std::setprecision(17);
        for (const auto& item : frames_) {
            auto affine = item.second.affine.detach().cpu();
            csv << item.first << ',' << affine[0].item<float>() << ',' << affine[1].item<float>() << '\n';
        }
    }
};
