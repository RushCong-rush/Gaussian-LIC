#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <torch/torch.h>
#include "rasterizer/rasterizer.h"
#include "rasterizer/cuda_rasterizer/config.h"
#include "rasterizer/cuda_rasterizer/rasterizer_impl.h"

namespace rasterizer_regression {

inline void checkAlphaSaturation()
{
    torch::AutoGradMode enable_grad(true);
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto background = torch::zeros({3}, options), identity = torch::eye(4, options);
    auto camera = torch::zeros({3}, options), empty = torch::empty({0}, options);
    auto means = torch::tensor({.05f, 0.f, 3.f}, options).reshape({1, 3});
    auto scales = torch::full({1, 3}, 1.3f, options);
    auto rotation = torch::tensor({1.f, 0.f, 0.f, 0.f}, options).reshape({1, 4});
    auto color = torch::tensor({.2f, .1f, .3f}, options).reshape({1, 1, 3});
    auto harmonics = torch::zeros({1, 15, 3}, options);
    GaussianRasterizationSettings settings(64, 128, 1, 1, -1, 1, -1, 1,
        background, 1, identity, identity, 0, camera, false, true, false, 0, true);
    GaussianRasterizer rasterizer(settings);
    auto loss = [&](const std::array<torch::Tensor, 3>& parameters) {
        auto result = rasterizer.forward(parameters[1], torch::zeros_like(means),
            parameters[0], parameters[2], harmonics, empty, scales, rotation, empty);
        return std::get<0>(result).index({0, 32, 64});
    };
    for (float opacity : {.8f, .998f})
    {
        std::array<torch::Tensor, 3> base{torch::full({1, 1}, opacity, options), means, color};
        std::array<torch::Tensor, 3> parameters;
        for (int i = 0; i < 3; ++i) parameters[i] = base[i].clone().requires_grad_();
        loss(parameters).backward();
        for (int variable = 0; variable < 3; ++variable)
        {
            torch::NoGradGuard guard;
            auto plus = base, minus = base;
            plus[variable] = base[variable].clone();
            minus[variable] = base[variable].clone();
            const float epsilon = variable == 0 ? .0001f : .001f;
            plus[variable].flatten()[0] += epsilon;
            minus[variable].flatten()[0] -= epsilon;
            const double numeric = (loss(plus).item<double>() - loss(minus).item<double>()) / (2 * epsilon);
            const double analytic = parameters[variable].grad().flatten()[0].item<double>();
            TORCH_CHECK(std::isfinite(analytic) &&
                std::abs(analytic - numeric) < .001 + .02 * std::abs(numeric),
                "Saturated/unsaturated alpha gradient disagrees with finite difference");
        }
    }
}

inline void checkScaleModifier()
{
    torch::AutoGradMode enable_grad(true);
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto background = torch::zeros({3}, options), identity = torch::eye(4, options);
    auto camera = torch::zeros({3}, options), empty = torch::empty({0}, options);
    auto means = torch::tensor({.3f, .2f, 3.f}, options).reshape({1, 3});
    auto base = torch::tensor({.3f, .5f, .2f}, options).reshape({1, 3});
    auto rotation = torch::tensor({1.f, 0.f, 0.f, 0.f}, options).reshape({1, 4});
    auto color = torch::tensor({.2f, .1f, .3f}, options).reshape({1, 1, 3});
    auto harmonics = torch::zeros({1, 15, 3}, options);
    auto opacity = torch::full({1, 1}, .8f, options);
    for (float modifier : {.5f, 1.f, 2.f})
    {
        GaussianRasterizationSettings settings(64, 128, 1, 1, -1, 1, -1, 1,
            background, modifier, identity, identity, 0, camera, false, true, false, 0, true);
        GaussianRasterizer rasterizer(settings);
        auto loss = [&](torch::Tensor scales) {
            auto result = rasterizer.forward(means, torch::zeros_like(means), opacity,
                color, harmonics, empty, scales, rotation, empty);
            return std::get<0>(result).index({0, 31, 67});
        };
        auto scales = base.clone().requires_grad_();
        loss(scales).backward();
        for (int axis = 0; axis < 3; ++axis)
        {
            torch::NoGradGuard guard;
            auto plus = base.clone(), minus = base.clone();
            constexpr float epsilon = .0005f;
            plus.flatten()[axis] += epsilon;
            minus.flatten()[axis] -= epsilon;
            const double numeric = (loss(plus).item<double>() - loss(minus).item<double>()) / (2 * epsilon);
            const double analytic = scales.grad().flatten()[axis].item<double>();
            TORCH_CHECK(std::isfinite(analytic) &&
                std::abs(analytic - numeric) < .001 + .02 * std::abs(numeric),
                "Scale-modifier chain rule disagrees with finite difference");
        }
    }
}

inline void checkCulledLanes()
{
    torch::NoGradGuard guard;
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto background = torch::zeros({3}, options), identity = torch::eye(4, options);
    auto camera = torch::zeros({3}, options), empty = torch::empty({0}, options);
    auto means = torch::tensor({.1f, .2f, 4.f}, options).reshape({1, 3}).repeat({37, 1});
    auto opacity = torch::full({37, 1}, .4f, options);
    auto indices = torch::arange(37, options.dtype(torch::kInt64));
    opacity.index_put_({indices.remainder(3) == 0}, 0.f);
    means.index_put_({indices.remainder(3) == 1}, 0.f);
    auto valid = indices.remainder(3) == 2;
    auto render = [&](bool erp, bool cache, bool depth_only, bool compact) {
        auto xyz = compact ? means.index({valid}) : means;
        auto alpha = compact ? opacity.index({valid}) : opacity;
        const auto n = xyz.size(0);
        auto rotation = torch::tensor({1.f, 0.f, 0.f, 0.f}, options).reshape({1, 4}).repeat({n, 1});
        return RasterizeGaussiansCUDA(background, xyz, empty, alpha,
            torch::full({n, 3}, .4f, options), rotation, 1.f, empty,
            identity, identity, 1.f, 1.f, 64, 128, -1.f, 1.f, -1.f, 1.f,
            torch::full({n, 1, 3}, .3f, options), empty, 0, camera,
            false, false, depth_only, erp, cache);
    };
    for (bool erp : {false, true})
    for (bool cache : {false, true})
    for (bool depth_only : {false, true})
    {
        auto full = render(erp, cache, depth_only, false);
        auto compact = render(erp, cache, depth_only, true);
        TORCH_CHECK(torch::equal(std::get<2>(full), std::get<2>(compact)) &&
            torch::equal(std::get<3>(full), std::get<3>(compact)) &&
            torch::equal(std::get<4>(full), std::get<4>(compact)),
            "Culled lanes changed RGB, transmittance or depth");
        TORCH_CHECK((std::get<5>(full).index({~valid}) == 0).all().item<bool>(),
            "Invisible Gaussians must have zero radius");
    }
}

inline void checkAllocationUnits()
{
    using namespace CudaRasterizer;
    auto span = [](const void* first, const void* next) {
        return reinterpret_cast<uintptr_t>(next) - reinterpret_cast<uintptr_t>(first);
    };
    for (size_t buckets : {size_t(1), size_t(33), size_t(100000)})
    {
        char* cursor = nullptr;
        const auto sample = SampleState::fromChunk(cursor, buckets);
        TORCH_CHECK(span(sample.bucket_to_tile, sample.T) <= sizeof(uint32_t) * buckets + 127,
            "bucket_to_tile must allocate per bucket, not per pixel");
    }
    const size_t pixels = 1024 * 512, tiles = pixels / (BLOCK_X * BLOCK_Y);
    char* cursor = nullptr;
    const auto image = ImageState::fromChunk(cursor, pixels, tiles);
    TORCH_CHECK(span(image.max_contrib, image.pixel_colors) <= sizeof(uint32_t) * tiles + 127 &&
        span(image.bucket_count, image.bucket_offsets) <= sizeof(uint32_t) * tiles + 127 &&
        span(image.bucket_offsets, image.bucket_count_scanning_space) <= sizeof(uint32_t) * tiles + 127,
        "Tile-indexed arrays must allocate per tile, not per pixel");
}

} // namespace rasterizer_regression
