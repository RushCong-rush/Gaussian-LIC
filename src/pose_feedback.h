#pragma once

#include "gaussian.h"
#include "rasterizer/rasterizer.h"
#include <gaussian_lic/RefinePose.h>

// Map tensors are detached; only the six-dimensional camera pose is optimized.
bool refineGaussianPose(const gaussian_lic::RefinePose::Request& request,
                        gaussian_lic::RefinePose::Response& response,
                        const std::shared_ptr<GaussianModel>& map,
                        const std::shared_ptr<Dataset>& dataset,
                        const Params& params);

namespace pose_feedback {
torch::Tensor quaternionProduct(const torch::Tensor& a, const torch::Tensor& b);
torch::Tensor worldColors(const torch::Tensor& xyz, const torch::Tensor& center,
                         const torch::Tensor& dc, const torch::Tensor& sh, int degree);
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> renderPose(
    GaussianRasterizer& rasterizer, const torch::Tensor& delta,
    const torch::Tensor& qcw, const torch::Tensor& twc,
    const torch::Tensor& xyz, const torch::Tensor& rotations,
    const torch::Tensor& scales, const torch::Tensor& opacity,
    const torch::Tensor& dc, const torch::Tensor& sh, int degree);
}
