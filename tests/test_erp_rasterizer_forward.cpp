#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tuple>

#include <torch/torch.h>

#include "rasterizer/rasterize_points.h"

namespace
{
constexpr int kHeight = 64;
constexpr int kWidth = 128;

struct RenderResult
{
    torch::Tensor alpha;
    torch::Tensor depth;
    torch::Tensor radii;
    int rendered;
};

RenderResult renderSingle(const float x, const float y, const float z,
                          const bool equirectangular, const bool no_color = false)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto background = torch::zeros({3}, options);
    auto means = torch::tensor({x, y, z}, options).reshape({1, 3});
    auto colors = torch::tensor({1.0f, 0.5f, 0.25f}, options).reshape({1, 3});
    auto opacity = torch::tensor({0.9f}, options);
    auto scales = torch::full({1, 3}, 0.20f, options);
    auto rotations = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}, options).reshape({1, 4});
    auto empty = torch::empty({0}, options);
    auto dc = torch::zeros({1, 1, 3}, options);
    auto view = torch::eye(4, options);
    auto projection = torch::eye(4, options);
    auto camera_position = torch::zeros({3}, options);

    auto result = RasterizeGaussiansCUDA(
        background, means, colors, opacity, scales, rotations, 1.0f, empty,
        view, projection, 1.0f, 1.0f, kHeight, kWidth,
        -1.0f, 1.0f, -1.0f, 1.0f, dc, empty, 0, camera_position,
        false, true, no_color, equirectangular);

    auto transmittance = std::get<3>(result).cpu();
    return {1.0f - transmittance, std::get<4>(result).cpu(), std::get<5>(result).cpu(), std::get<0>(result)};
}

void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::pair<int, int> peakPixel(const torch::Tensor& image)
{
    const int64_t index = image.argmax().item<int64_t>();
    return {static_cast<int>(index % kWidth), static_cast<int>(index / kWidth)};
}

void requireNear(const int actual, const int expected, const int tolerance, const char* message)
{
    if (std::abs(actual - expected) > tolerance)
    {
        std::cerr << "FAILED: " << message << " (actual=" << actual
                  << ", expected=" << expected << ", tolerance=" << tolerance << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

void savePgm(const std::filesystem::path& path, torch::Tensor image, const float scale = 1.0f)
{
    image = (image * scale).clamp(0.0f, 1.0f).mul(255.0f).to(torch::kUInt8).contiguous();
    std::ofstream stream(path, std::ios::binary);
    stream << "P5\n" << image.size(1) << ' ' << image.size(0) << "\n255\n";
    stream.write(reinterpret_cast<const char*>(image.data_ptr<uint8_t>()), image.numel());
}

using ForwardResult = std::tuple<
    int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
using BackwardResult = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

struct GradientFixture
{
    torch::Tensor background;
    torch::Tensor colors;
    torch::Tensor opacity;
    torch::Tensor empty;
    torch::Tensor dc;
    torch::Tensor view;
    torch::Tensor projection;
    torch::Tensor camera_position;

    GradientFixture()
    {
        const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        background = torch::zeros({3}, options);
        colors = torch::tensor({0.8f, 0.35f, 0.15f}, options).reshape({1, 3});
        opacity = torch::tensor({0.82f}, options);
        empty = torch::empty({0}, options);
        dc = torch::zeros({1, 1, 3}, options);
        view = torch::eye(4, options);
        projection = torch::eye(4, options);
        camera_position = torch::zeros({3}, options);
    }

    ForwardResult forward(
        const torch::Tensor& means, const torch::Tensor& scales,
        const torch::Tensor& rotations, const bool debug = false) const
    {
        return RasterizeGaussiansCUDA(
            background, means, colors, opacity, scales, rotations, 1.0f, empty,
            view, projection, 1.0f, 1.0f, kHeight, kWidth,
            -1.0f, 1.0f, -1.0f, 1.0f, dc, empty, 0, camera_position,
            false, debug, false, true);
    }

    float loss(
        const torch::Tensor& means, const torch::Tensor& scales,
        const torch::Tensor& rotations, const torch::Tensor& color_gradient,
        const torch::Tensor& depth_gradient) const
    {
        const auto result = forward(means, scales, rotations);
        return (std::get<2>(result) * color_gradient).sum().item<float>()
            + (std::get<4>(result) * depth_gradient).sum().item<float>();
    }

    BackwardResult backward(
        const torch::Tensor& means, const torch::Tensor& scales,
        const torch::Tensor& rotations, const torch::Tensor& color_gradient,
        const torch::Tensor& depth_gradient) const
    {
        const auto result = forward(means, scales, rotations, true);
        return RasterizeGaussiansBackwardCUDA(
            background, means, std::get<5>(result), colors, scales, rotations, 1.0f,
            empty, view, projection, 1.0f, 1.0f,
            -1.0f, 1.0f, -1.0f, 1.0f,
            color_gradient, depth_gradient, dc, empty, 0, camera_position,
            std::get<6>(result), std::get<0>(result), std::get<7>(result),
            std::get<8>(result), std::get<1>(result), std::get<9>(result),
            0.0f, true, true);
    }
};

enum class GradientParameter
{
    Mean,
    Scale,
    Rotation
};

float finiteDifference(
    const GradientFixture& fixture,
    const torch::Tensor& means, const torch::Tensor& scales, const torch::Tensor& rotations,
    const torch::Tensor& color_gradient, const torch::Tensor& depth_gradient,
    const GradientParameter parameter, const int component, const float epsilon)
{
    auto means_positive = means.clone();
    auto means_negative = means.clone();
    auto scales_positive = scales.clone();
    auto scales_negative = scales.clone();
    auto rotations_positive = rotations.clone();
    auto rotations_negative = rotations.clone();

    torch::Tensor* positive = nullptr;
    torch::Tensor* negative = nullptr;
    if (parameter == GradientParameter::Mean)
    {
        positive = &means_positive;
        negative = &means_negative;
    }
    else if (parameter == GradientParameter::Scale)
    {
        positive = &scales_positive;
        negative = &scales_negative;
    }
    else
    {
        positive = &rotations_positive;
        negative = &rotations_negative;
    }

    const float value = positive->index({0, component}).item<float>();
    positive->index_put_({0, component}, value + epsilon);
    negative->index_put_({0, component}, value - epsilon);
    const float loss_positive = fixture.loss(
        means_positive, scales_positive, rotations_positive, color_gradient, depth_gradient);
    const float loss_negative = fixture.loss(
        means_negative, scales_negative, rotations_negative, color_gradient, depth_gradient);
    return (loss_positive - loss_negative) / (2.0f * epsilon);
}

void requireGradientClose(
    const char* label, const int component, const float analytic, const float numeric,
    const float relative_tolerance = 0.08f, const float absolute_tolerance = 5.0e-4f)
{
    const float tolerance = absolute_tolerance
        + relative_tolerance * std::max(std::abs(analytic), std::abs(numeric));
    if (!std::isfinite(analytic) || !std::isfinite(numeric) || std::abs(analytic - numeric) > tolerance)
    {
        std::cerr << "FAILED: " << label << '[' << component << "] analytic=" << analytic
                  << ", numeric=" << numeric << ", tolerance=" << tolerance << '\n';
        std::exit(EXIT_FAILURE);
    }
}

torch::Tensor makeColorGradient(const torch::TensorOptions& options)
{
    auto horizontal = torch::linspace(-1.0f, 1.0f, kWidth, options).view({1, kWidth}).repeat({kHeight, 1});
    auto vertical = torch::linspace(-1.0f, 1.0f, kHeight, options).view({kHeight, 1}).repeat({1, kWidth});
    return torch::stack({
        0.4f + 0.3f * horizontal + 0.1f * vertical,
        -0.2f + 0.15f * vertical,
        0.25f - 0.1f * horizontal * vertical}) / 256.0f;
}

void checkErpGradients()
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const GradientFixture fixture;
    const auto scales = torch::tensor({0.25f, 0.17f, 0.12f}, options).reshape({1, 3});
    const auto rotations = torch::tensor({0.97f, 0.10f, -0.08f, 0.04f}, options).reshape({1, 4});
    const auto color_gradient = makeColorGradient(options);
    const auto no_depth_gradient = torch::zeros({kHeight, kWidth}, options);

    const auto means = torch::tensor({0.8f, -0.4f, 4.0f}, options).reshape({1, 3});
    const auto analytic = fixture.backward(means, scales, rotations, color_gradient, no_depth_gradient);
    const auto analytic_means = std::get<3>(analytic).cpu();
    const auto analytic_scales = std::get<7>(analytic).cpu();
    const auto analytic_rotations = std::get<8>(analytic).cpu();
    for (int component = 0; component < 3; ++component)
    {
        const float regular_numeric_mean = finiteDifference(
            fixture, means, scales, rotations, color_gradient, no_depth_gradient,
            GradientParameter::Mean, component, 1.0e-2f);
        requireGradientClose(
            "regular mean", component, analytic_means.index({0, component}).item<float>(),
            regular_numeric_mean);
        requireGradientClose(
            "regular scale", component, analytic_scales.index({0, component}).item<float>(),
            finiteDifference(fixture, means, scales, rotations, color_gradient, no_depth_gradient,
                             GradientParameter::Scale, component, 1.0e-3f));
    }
    for (int component = 0; component < 4; ++component)
    {
        requireGradientClose(
            "regular rotation", component, analytic_rotations.index({0, component}).item<float>(),
            finiteDifference(fixture, means, scales, rotations, color_gradient, no_depth_gradient,
                             GradientParameter::Rotation, component, 1.0e-3f), 0.10f, 1.0e-3f);
    }

    const auto seam_means = torch::tensor({0.12f, 0.15f, -4.0f}, options).reshape({1, 3});
    const auto seam_analytic = fixture.backward(
        seam_means, scales, rotations, color_gradient, no_depth_gradient);
    const auto seam_mean_gradient = std::get<3>(seam_analytic).cpu();
    for (int component = 0; component < 3; ++component)
    {
        requireGradientClose(
            "seam mean", component, seam_mean_gradient.index({0, component}).item<float>(),
            finiteDifference(fixture, seam_means, scales, rotations, color_gradient, no_depth_gradient,
                             GradientParameter::Mean, component, 1.0e-2f), 0.10f, 1.0e-3f);
    }

    auto center_depth_gradient = torch::zeros({kHeight, kWidth}, options);
    center_depth_gradient.index_put_({kHeight / 2 + 2, kWidth / 2 + 4}, 1.0f);
    const auto depth_analytic = fixture.backward(
        means, scales, rotations, torch::zeros_like(color_gradient), center_depth_gradient);
    const auto depth_mean_gradient = std::get<3>(depth_analytic).cpu();
    for (int component = 0; component < 3; ++component)
    {
        requireGradientClose(
            "radial depth mean", component, depth_mean_gradient.index({0, component}).item<float>(),
            finiteDifference(fixture, means, scales, rotations, torch::zeros_like(color_gradient),
                             center_depth_gradient, GradientParameter::Mean, component, 1.0e-2f),
            0.08f, 1.0e-3f);
    }
}
}

void checkMixedSeamTileCounts()
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto means = torch::tensor({{0.1f, 0.5f, -5.0f}, {0.0f, 0.2f, 5.0f}}, options);
    auto tileCount = [&](const torch::Tensor& points)
    {
        const auto n = points.size(0);
        auto empty = torch::empty({0}, options);
        auto rotations = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}, options).repeat({n, 1});
        auto result = RasterizeGaussiansCUDA(
            torch::zeros({3}, options), points.contiguous(), empty,
            torch::full({n, 1}, 0.8f, options), torch::full({n, 3}, 1.5f, options),
            rotations, 1.0f, empty, torch::eye(4, options), torch::eye(4, options),
            1.0f, 1.0f, 512, 1024, -1.0f, 1.0f, -1.0f, 1.0f,
            torch::zeros({n, 1, 3}, options), empty, 0, torch::zeros({3}, options),
            false, true, true, true);
        return std::get<0>(result);
    };
    // Tile counts must be additive even when only some lanes cross the seam.
    const int expected = tileCount(means.narrow(0, 0, 1)) + tileCount(means.narrow(0, 1, 1));
    require(expected > 64, "mixed-seam fixture must exercise cooperative tile counting");
    require(tileCount(means) == expected, "mixed-seam warp lost Gaussian-tile pairs");
    require(tileCount(means.flip({0})) == expected, "tile count depends on Gaussian ordering");
}

int main(int argc, char** argv)
{
    torch::NoGradGuard no_grad;
    require(torch::cuda::is_available(), "CUDA is not available");

    const auto forward = renderSingle(0.0f, 0.0f, 4.0f, true);
    const auto right = renderSingle(4.0f, 0.0f, 0.0f, true);
    const auto left = renderSingle(-4.0f, 0.0f, 0.0f, true);
    const auto seam = renderSingle(0.0f, 0.0f, -4.0f, true);

    const auto forward_peak = peakPixel(forward.alpha);
    const auto right_peak = peakPixel(right.alpha);
    const auto left_peak = peakPixel(left.alpha);
    requireNear(forward_peak.first, kWidth / 2, 1, "+Z longitude projection is incorrect");
    requireNear(forward_peak.second, kHeight / 2, 1, "+Z latitude projection is incorrect");
    requireNear(right_peak.first, 3 * kWidth / 4, 1, "+X longitude projection is incorrect");
    requireNear(left_peak.first, kWidth / 4, 1, "-X longitude projection is incorrect");

    constexpr float pi = 3.14159265358979323846f;
    const float angular_pixel_scale = kWidth / (2.0f * pi);
    const float projected_standard_deviation = 0.20f * angular_pixel_scale / 4.0f;
    const float reference_variance = projected_standard_deviation * projected_standard_deviation + 0.3f;
    const int reference_radius = static_cast<int>(std::ceil(3.0f * std::sqrt(reference_variance)));
    require(forward.radii.index({0}).item<int>() == reference_radius,
            "single-Gaussian ERP covariance disagrees with the CPU reference");

    require(seam.alpha.index({torch::indexing::Slice(), 0}).max().item<float>() > 0.01f,
            "seam Gaussian does not cover the left image edge");
    require(seam.alpha.index({torch::indexing::Slice(), kWidth - 1}).max().item<float>() > 0.01f,
            "seam Gaussian does not cover the right image edge");

    const float center_depth = forward.depth.index({kHeight / 2, kWidth / 2}).item<float>();
    require(std::abs(center_depth - 4.0f) < 1.0e-3f, "ERP depth is not radial distance");
    const auto no_color_forward = renderSingle(0.0f, 0.0f, 4.0f, true, true);
    require(torch::allclose(no_color_forward.alpha, forward.alpha),
            "no-color ERP alpha differs from the full render");
    require(torch::allclose(no_color_forward.depth, forward.depth),
            "no-color ERP depth differs from the full render");

    const auto pole = renderSingle(0.0f, -4.0f, 1.0e-3f, true);
    require(torch::isfinite(pole.alpha).all().item<bool>(), "near-pole alpha contains NaN or Inf");
    require(torch::isfinite(pole.depth).all().item<bool>(), "near-pole depth contains NaN or Inf");
    require(pole.alpha.max().item<float>() > 0.01f, "near-pole Gaussian is not rendered");

    const auto pinhole = renderSingle(0.0f, 0.0f, 4.0f, false);
    const auto pinhole_peak = peakPixel(pinhole.alpha);
    if (pinhole.alpha.max().item<float>() <= 0.01f)
    {
        std::cerr << "pinhole diagnostics: rendered=" << pinhole.rendered
                  << ", radius=" << pinhole.radii.index({0}).item<int>() << '\n';
    }
    require(pinhole.alpha.max().item<float>() > 0.01f, "pinhole Gaussian is not rendered");
    requireNear(pinhole_peak.first, kWidth / 2, 1, "pinhole horizontal projection regressed");
    requireNear(pinhole_peak.second, kHeight / 2, 1, "pinhole vertical projection regressed");
    require(std::abs(pinhole.depth.index({kHeight / 2, kWidth / 2}).item<float>() - 4.0f) < 1.0e-3f,
            "pinhole z-depth regressed");

    checkErpGradients();
    checkMixedSeamTileCounts();

    if (argc == 2)
    {
        const std::filesystem::path output_dir(argv[1]);
        std::filesystem::create_directories(output_dir);
        auto first_row = torch::cat({left.alpha, forward.alpha, right.alpha}, 1);
        auto second_row = torch::cat({seam.alpha, pole.alpha, pinhole.alpha}, 1);
        savePgm(output_dir / "erp_forward_montage.pgm", torch::cat({first_row, second_row}, 0));
        savePgm(output_dir / "erp_seam_alpha.pgm", seam.alpha);
        savePgm(output_dir / "erp_radial_depth.pgm", forward.depth, 0.25f);
    }

    std::cout << "erp_rasterizer_forward_test passed\n";
    return EXIT_SUCCESS;
}
