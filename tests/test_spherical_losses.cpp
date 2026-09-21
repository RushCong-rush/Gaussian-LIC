#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include "loss_utils.h"
#include "depth_loss.h"

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    torch::manual_seed(17);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto target = torch::rand({1, 3, 31, 64}, options);
    auto input = (target * .8f + .05f).clone().requires_grad_();
    auto mask = torch::rand({31, 64}, options) > .2f;
    auto original = loss_utils::fused_ssim_erp(input, target, mask);
    original.backward();
    auto grad = input.grad().clone();
    auto rolled = torch::roll(input.detach(), {23}, {3}).requires_grad_();
    auto shifted = loss_utils::fused_ssim_erp(rolled, torch::roll(target, {23}, {3}),
                                            torch::roll(mask, {23}, {1}));
    shifted.backward();
    require(std::abs(original.item<float>() - shifted.item<float>()) < 2.e-6,
            "periodic SSIM changed under longitude rotation");
    require(torch::allclose(torch::roll(grad, {23}, {3}), rolled.grad(), 3.e-3, 2.e-6),
            "periodic SSIM gradient changed under longitude rotation");
    require(grad.abs().sum().item<float>() > .01f, "SSIM gradient vanished");
    auto same = loss_utils::fused_ssim_erp(target, target);
    require(std::abs(same.item<float>() - 1.f) < 1.e-5, "identity SSIM is not one");
    // A boundary pixel checks gradient accumulation through the wrapped copies.
    auto plus = input.detach().clone(), minus = input.detach().clone();
    plus.index_put_({0, 1, 15, 0}, plus.index({0,1,15,0}) + .01f);
    minus.index_put_({0, 1, 15, 0}, minus.index({0,1,15,0}) - .01f);
    float numeric = (loss_utils::fused_ssim_erp(plus,target,mask).item<float>() -
                     loss_utils::fused_ssim_erp(minus,target,mask).item<float>()) / .02f;
    float analytic = grad.index({0,1,15,0}).item<float>();
    std::cout << "Boundary SSIM gradient: " << analytic << " vs " << numeric << '\n';
    require(std::abs(numeric-analytic) < 1.e-5, "boundary SSIM gradient failed finite difference");
    auto weights = loss_utils::latitude_weights(8, 64, options);
    auto ones = torch::ones({3, 8, 64}, options).requires_grad_();
    auto zero = torch::zeros_like(ones);
    auto weighted_l1 = loss_utils::weighted_l1_loss(ones, zero, weights);
    require(std::abs(weighted_l1.item<float>() - weights.mean().item<float>()) < 1.e-6,
            "latitude L1 changed the valid-pixel normalization");
    weighted_l1.backward();
    require(ones.grad().index({0,0,0}).abs().item<float>() < 1.e-9,
            "polar RGB error was not downweighted");
    require(ones.grad().index({0,4,0}).item<float>() > .0001f,
            "equatorial RGB gradient vanished");
    auto depth = torch::ones({8,64}, options).requires_grad_();
    auto depth_mask = torch::ones({8,64}, options.dtype(torch::kBool));
    auto loss_depth = loss_utils::maskedDepthL1(depth, torch::zeros_like(depth), depth_mask, weights);
    require(std::abs(loss_depth.item<float>() - weights.mean().item<float>()) < 1.e-6,
            "latitude weights were not applied to depth errors");
    auto empty = torch::zeros_like(depth_mask);
    require(loss_utils::maskedDepthL1(depth,depth,empty,weights).item<float>() == 0,
            "empty sparse depth support changed behavior");
    auto lat = loss_utils::latitude_weights(31,64,options);
    auto weighted_ssim = loss_utils::fused_ssim_erp(target,target,mask,lat);
    require(std::abs(weighted_ssim.item<float>()-1.f) < 1.e-5,
            "perfect weighted SSIM should have zero loss");
    auto ws1 = loss_utils::fused_ssim_erp(input,target,mask,lat);
    auto ws2 = loss_utils::fused_ssim_erp(torch::roll(input,{23},{3}),
        torch::roll(target,{23},{3}),torch::roll(mask,{23},{1}),lat);
    require(std::abs(ws1.item<float>()-ws2.item<float>()) < 2.e-6,
            "latitude-weighted SSIM broke longitude invariance");
    std::cout << "spherical_loss_test passed\n";
}
