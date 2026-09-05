#include <cmath>
#include <cstdlib>
#include <iostream>

#include "loss_utils.h"

namespace
{
void requireNear(double actual, double expected, const char* message)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-6)
    {
        std::cerr << "FAILED: " << message << " (actual=" << actual
                  << ", expected=" << expected << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

void checkSourceWeights(torch::Device device, int dap_count, double weight)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto rendered = torch::full({dap_count + 3}, 3.0f, options).set_requires_grad(true);
    auto target = torch::ones_like(rendered);
    auto valid = torch::ones_like(rendered, torch::kBool);
    valid.index_put_({dap_count + 2}, false);
    auto lidar = torch::zeros_like(valid);
    lidar.slice(0, 0, 2).fill_(true);
    auto loss = loss_utils::source_balanced_depth_l1(rendered, target, valid, lidar, weight);
    requireNear(loss.item<double>(), 2.0 + (dap_count ? 2.0 * weight : 0.0), "source loss");
    loss.backward();
    auto grad = rendered.grad();
    requireNear(grad.slice(0, 0, 2).sum().item<double>(), 1.0, "LiDAR gradient not diluted");
    requireNear(grad.slice(0, 2, dap_count + 2).sum().item<double>(),
                dap_count ? weight : 0.0, "DAP-only gradient weight");
    requireNear(grad.index({dap_count + 2}).item<double>(), 0.0, "invalid pixel excluded");

    if (weight == 0.0)
    {
        auto baseline = torch::abs(rendered.detach().masked_select(valid & lidar)
                                   - target.masked_select(valid & lidar)).mean();
        requireNear(loss.item<double>(), baseline.item<double>(), "zero DAP weight equals LiDAR-only");
    }
}

void checkEmptyLidar(torch::Device device, bool has_valid)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto rendered = torch::full({4}, 3.0f, options).set_requires_grad(true);
    auto target = torch::ones_like(rendered);
    auto valid = torch::full_like(rendered, has_valid, torch::kBool);
    auto lidar = torch::zeros_like(valid);
    auto loss = loss_utils::source_balanced_depth_l1(rendered, target, valid, lidar, 0.1);
    requireNear(loss.item<double>(), has_valid ? 0.2 : 0.0, "empty LiDAR source");
    loss.backward();
    requireNear(rendered.grad().sum().item<double>(), has_valid ? 0.1 : 0.0, "empty source gradients");
}
}

int main()
{
    for (auto device : {torch::Device(torch::kCPU), torch::Device(torch::kCUDA)})
    {
        for (int count : {0, 1, 1000})
            for (double weight : {0.0, 0.1, 1.0})
                checkSourceWeights(device, count, weight);
        checkEmptyLidar(device, false);
        checkEmptyLidar(device, true);
    }
    std::cout << "Depth supervision value/gradient tests passed on CPU and CUDA\n";
}
