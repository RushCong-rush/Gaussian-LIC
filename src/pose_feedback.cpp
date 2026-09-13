#include "pose_feedback.h"
#include "general_utils.h"
#include "loss_utils.h"
#include "rasterizer/rasterizer.h"

namespace pose_feedback {
using torch::indexing::Slice;

torch::Tensor quaternionProduct(const torch::Tensor& a, const torch::Tensor& b)
{
    auto aw = a.index({Slice(), Slice(0, 1)});
    auto av = a.index({Slice(), Slice(1, 4)});
    auto bw = b.index({Slice(), Slice(0, 1)});
    auto bv = b.index({Slice(), Slice(1, 4)});
    return torch::cat({aw*bw - (av*bv).sum(1, true),
                      aw*bv + bw*av + torch::cross(av.expand_as(bv), bv, 1)}, 1);
}

torch::Tensor worldColors(const torch::Tensor& xyz, const torch::Tensor& center,
                         const torch::Tensor& dc, const torch::Tensor& sh, int degree)
{
    auto direction = xyz-center;
    direction = direction / direction.norm(2, 1, true);
    auto x=direction.select(1,0).unsqueeze(1), y=direction.select(1,1).unsqueeze(1);
    auto z=direction.select(1,2).unsqueeze(1);
    auto c=0.28209479177387814*dc.squeeze(1);
    if (degree>0)
        c=c+0.4886025119029199*(-y*sh.select(1,0)+z*sh.select(1,1)-x*sh.select(1,2));
    auto xx=x*x, yy=y*y, zz=z*z;
    if (degree>1)
        c=c+1.0925484305920792*x*y*sh.select(1,3)
          -1.0925484305920792*y*z*sh.select(1,4)
          +0.31539156525252005*(2*zz-xx-yy)*sh.select(1,5)
          -1.0925484305920792*x*z*sh.select(1,6)
          +0.5462742152960396*(xx-yy)*sh.select(1,7);
    if (degree>2)
        c=c-0.5900435899266435*y*(3*xx-yy)*sh.select(1,8)
          +2.890611442640554*x*y*z*sh.select(1,9)
          -0.4570457994644658*y*(4*zz-xx-yy)*sh.select(1,10)
          +0.3731763325901154*z*(2*zz-3*xx-3*yy)*sh.select(1,11)
          -0.4570457994644658*x*(4*zz-xx-yy)*sh.select(1,12)
          +1.445305721320277*z*(xx-yy)*sh.select(1,13)
          -0.5900435899266435*x*(xx-3*yy)*sh.select(1,14);
    return (c+0.5).clamp_min(0);
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor> renderPose(
    GaussianRasterizer& rasterizer, const torch::Tensor& delta,
    const torch::Tensor& qcw, const torch::Tensor& twc,
    const torch::Tensor& xyz, const torch::Tensor& rotations,
    const torch::Tensor& scales, const torch::Tensor& opacity,
    const torch::Tensor& dc, const torch::Tensor& sh, int degree)
{
    auto dq=torch::cat({torch::ones({1},delta.options().requires_grad(false)),
                       delta.index({Slice(3,6)})*.5}).reshape({1,4});
    dq=dq/dq.norm(2,1,true);
    auto qc=quaternionProduct(dq,qcw);
    auto qr=qc.clone();
    auto rcw=general_utils::build_rotation(qr).squeeze(0);
    auto center=twc+delta.index({Slice(0,3)});
    auto means=torch::matmul(xyz-center,rcw.transpose(0,1)).contiguous();
    auto rot=quaternionProduct(qc,rotations).contiguous();
    // World-frame SH colors travel through the existing degree-zero DC path.
    auto color_dc=((worldColors(xyz,center,dc,sh,degree)-.5)
                   /0.28209479177387814).unsqueeze(1).contiguous();
    return rasterizer.forward(means,torch::zeros_like(means),opacity,color_dc,
        sh,torch::Tensor(),scales,rot,torch::Tensor());
}
}

bool refineGaussianPose(const gaussian_lic::RefinePose::Request& req,
                        gaussian_lic::RefinePose::Response& res,
                        const std::shared_ptr<GaussianModel>& map,
                        const std::shared_ptr<Dataset>& dataset,
                        const Params& prm)
{
    using torch::indexing::Slice;
    res.valid=false;
    res.pose=req.initial_pose;
    if (!req.optimize || !map->is_init_) return true;
    TORCH_CHECK(prm.equirectangular, "Gaussian pose feedback currently requires ERP");
    cv::Mat rgb;
    cv::cvtColor(cv_bridge::toCvCopy(req.image, "bgr8")->image, rgb, cv::COLOR_BGR2RGB);
    auto target=torch::from_blob(rgb.data,{rgb.rows,rgb.cols,3},torch::kUInt8)
        .to(torch::kCUDA).to(torch::kFloat32).permute({2,0,1})/255.f;
    const auto& p=req.initial_pose;
    Eigen::Quaterniond qwc(p.orientation.w,p.orientation.x,p.orientation.y,p.orientation.z);
    Eigen::Vector3d twc(p.position.x,p.position.y,p.position.z);
    auto opts=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto q0=torch::tensor({float(qwc.w()),float(-qwc.x()),float(-qwc.y()),float(-qwc.z())},opts).reshape({1,4});
    auto t0=torch::tensor({float(twc.x()),float(twc.y()),float(twc.z())},opts);
    auto delta=torch::zeros({6},opts.requires_grad(true));
    auto xyz=map->getXYZ().detach(), scales=map->getScaling().detach();
    auto rotations=map->getRotation().detach(), opacity=map->getOpacity().detach();
    auto dc=map->getFeaturesDc().detach(), sh=map->getFeaturesRest().detach();
    auto eye=torch::eye(4,opts), zero=torch::zeros({3},opts);
    auto background=prm.white_background ? torch::ones({3},opts) : zero;
    GaussianRasterizationSettings settings(rgb.rows,rgb.cols,1,1,-1,1,-1,1,
        background,1,eye,eye,0,zero,false,false,false,0,true);
    settings.valid_mask_=dataset->raster_mask_;
    GaussianRasterizer rasterizer(settings);
    auto valid=dataset->metric_mask_.defined() ? dataset->metric_mask_.to(torch::kCUDA)
                                             : torch::ones({rgb.rows,rgb.cols},opts.dtype(torch::kBool));
    auto gray=target.mean(0);
    auto gx=(gray.roll({-1},{1})-gray.roll({1},{1})).abs();
    auto gy=(gray.roll({-1},{0})-gray.roll({1},{0})).abs();
    auto support=valid & ((gx+gy)>0.02);
    auto padded_valid=torch::cat({valid.index({Slice(),Slice(-5,torch::indexing::None)}),
                                 valid,valid.index({Slice(),Slice(0,5)})},1).to(torch::kFloat32);
    auto neighborhood=torch::nn::functional::avg_pool2d(padded_valid.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::AvgPool2dFuncOptions(11).stride(1).padding(std::vector<int64_t>{5,0}));
    support=support & (neighborhood.squeeze()>.9999);
    auto render_pose=[&]() {
        return pose_feedback::renderPose(rasterizer,delta,q0,t0,xyz,rotations,
                                        scales,opacity,dc,sh,map->sh_degree_);
    };
    auto first=render_pose();
    support=support & ((1-std::get<3>(first).detach().squeeze())>0.9);
    const int pixels=support.sum().item<int>();
    if (pixels==0) return true;
    auto loss_for=[&](const torch::Tensor& image) {
        auto l1=(image-target).abs().masked_select(support.unsqueeze(0).expand_as(image)).mean();
        // Wrap horizontal SSIM neighborhoods and exclude invalid neighborhoods.
        auto a=image.unsqueeze(0), b=target.unsqueeze(0);
        a=torch::cat({a.index({Slice(),Slice(),Slice(),Slice(-5,torch::indexing::None)}),a,
                      a.index({Slice(),Slice(),Slice(),Slice(0,5)})},3).contiguous();
        b=torch::cat({b.index({Slice(),Slice(),Slice(),Slice(-5,torch::indexing::None)}),b,
                      b.index({Slice(),Slice(),Slice(),Slice(0,5)})},3).contiguous();
        auto mask=torch::cat({torch::zeros({rgb.rows,5},opts.dtype(torch::kBool)),support,
                             torch::zeros({rgb.rows,5},opts.dtype(torch::kBool))},1);
        return .5*l1+.5*(1-loss_utils::fused_ssim_masked(a,b,mask));
    };
    auto initial_loss=loss_for(std::get<0>(first));
    res.loss_before=initial_loss.item<double>();
    torch::optim::Adam optimizer({delta},torch::optim::AdamOptions(0.001));
    for (int i=0;i<20;++i)
    {
        optimizer.zero_grad();
        auto pkg=i==0 ? first : render_pose();
        auto loss=i==0 ? initial_loss : loss_for(std::get<0>(pkg));
        loss.backward();
        optimizer.step();
    }
    torch::NoGradGuard no_grad;
    res.loss_after=loss_for(std::get<0>(render_pose())).item<double>();
    if (!std::isfinite(res.loss_after) || res.loss_after>=res.loss_before) return true;
    auto d=delta.detach().cpu().contiguous();
    const float* v=d.data_ptr<float>();
    Eigen::Quaterniond dq(1,v[3]*.5,v[4]*.5,v[5]*.5);
    Eigen::Quaterniond refined=(dq.normalized()*qwc.conjugate()).conjugate();
    res.pose.position.x+=v[0]; res.pose.position.y+=v[1]; res.pose.position.z+=v[2];
    res.pose.orientation.w=refined.w(); res.pose.orientation.x=refined.x();
    res.pose.orientation.y=refined.y(); res.pose.orientation.z=refined.z();
    res.valid=true;
    return true;
}
