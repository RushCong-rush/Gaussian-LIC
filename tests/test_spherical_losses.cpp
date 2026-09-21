#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include "loss_utils.h"

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
    std::cout << "spherical_loss_test passed\n";
}
