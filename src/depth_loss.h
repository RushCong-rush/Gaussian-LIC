#pragma once

#include <torch/torch.h>

namespace loss_utils
{
inline torch::Tensor maskedDepthL1(const torch::Tensor& rendered,
                                   const torch::Tensor& reference,
                                   const torch::Tensor& mask,
                                   const torch::Tensor& latitude_weights = torch::Tensor())
{
    auto errors = (rendered.masked_select(mask) - reference.masked_select(mask)).abs();
    if (latitude_weights.defined())
        errors = errors * latitude_weights.expand_as(rendered).masked_select(mask);
    return errors.numel() == 0 ? errors.sum() : errors.mean();
}

inline torch::Tensor sourceBalancedDepthL1(const torch::Tensor& rendered,
                                          const torch::Tensor& reference,
                                          const torch::Tensor& lidar_mask,
                                          const torch::Tensor& valid_mask,
                                          double dap_relative_weight,
                                          const torch::Tensor& latitude_weights = torch::Tensor())
{
    return maskedDepthL1(rendered, reference, valid_mask & lidar_mask, latitude_weights)
        + dap_relative_weight * maskedDepthL1(rendered, reference, valid_mask & ~lidar_mask, latitude_weights);
}
}
