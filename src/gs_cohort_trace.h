#pragma once
#ifdef ODGS_GS_TRACE
#include <torch/csrc/autograd/autograd.h>
extern "C" void gsTracePointers(const unsigned char*,const bool*,float*,int);
namespace gs_trace {
inline bool enabled() {return std::getenv("ODGS_COHORT_TRACE")!=nullptr;}
struct Step {
    bool active=false;
    torch::Tensor ids,selection,pixels,raster,labels,shape;
    std::vector<torch::Tensor> photo,total,before;
};
inline bool sample(int frame,int view,int step) {
    const std::vector<int> frames={879,884,889,894,899,904,914,934,949,999,1049};
    return enabled()&&std::find(frames.begin(),frames.end(),frame)!=frames.end()&&
        (view==frame||view==899||step==0);
}
inline Step begin(const std::shared_ptr<Dataset>& d,std::shared_ptr<GaussianModel> pc,int view,int step) {
    Step s;if(!sample(d->all_frame_num_-1,view,step))return s;
    torch::NoGradGuard guard;auto lab=daeo::metadata(pc);
    auto keep=(lab.select(1,1)>=879)&(lab.select(1,1)<=899)&(lab.select(1,2)==0);
    s.ids=torch::nonzero(keep).squeeze(1).cuda();if(!s.ids.numel())return s;
    s.active=true;s.labels=lab.index({keep});s.selection=keep.to(torch::kCUDA).to(torch::kUInt8);
    s.pixels=d->metric_mask_.to(torch::kCUDA).to(torch::kBool).contiguous();
    s.raster=torch::zeros({8,pc->getXYZ().size(0)},pc->getXYZ().options().requires_grad(false));
    s.shape=torch::stack({std::get<0>(pc->getScaling().max(1)),pc->getOpacity().flatten()},1).index_select(0,s.ids).detach().cpu();
    gsTracePointers(s.selection.data_ptr<unsigned char>(),s.pixels.data_ptr<bool>(),s.raster.data_ptr<float>(),pc->getXYZ().size(0));return s;
}
inline void endRender(Step& s) {if(s.active){torch::cuda::synchronize();gsTracePointers(nullptr,nullptr,nullptr,0);}}
template<class Package>
inline void verify(Step& s,const std::shared_ptr<Dataset>& d,std::shared_ptr<GaussianModel> pc,
                   std::shared_ptr<Camera> cam,torch::Tensor bg,Package& original) {
    if(!s.active||d->all_frame_num_-1!=879||cam->frame_index_!=879)return;
    torch::NoGradGuard guard;
    auto all=torch::ones_like(s.selection);auto values=torch::zeros_like(s.raster);
    gsTracePointers(all.data_ptr<unsigned char>(),s.pixels.data_ptr<bool>(),values.data_ptr<float>(),pc->getXYZ().size(0));
    auto check=render(cam,pc,bg,pc->apply_exposure_,false,1.f,true,false);
    torch::cuda::synchronize();gsTracePointers(nullptr,nullptr,nullptr,0);
    float rgb_error=(std::get<0>(check)-std::get<0>(original)).abs().max().template item<float>();
    float depth_error=(std::get<1>(check)-std::get<1>(original)).abs().max().template item<float>();
    auto rgb=(std::get<0>(original)*s.pixels).flatten(1).sum(1);
    auto contributed=values.index({torch::indexing::Slice(3,6)}).sum(1);
    float sum_relative=((contributed-rgb).abs()/rgb.abs().clamp_min(1)).max().template item<float>();
    auto alpha=((1-std::get<2>(original))*s.pixels).sum();
    float alpha_relative=((values.select(0,2).sum()-alpha).abs()/alpha.clamp_min(1)).template item<float>();
    TORCH_CHECK(rgb_error<1e-5&&depth_error<1e-5&&sum_relative<1e-4&&alpha_relative<1e-4,"GS trace collector verification failed");
    std::ofstream f(daeo::folder(d)+"/cohort_trace_verification.json");
    f<<"{\"rgb_max_error\":"<<rgb_error<<",\"depth_max_error\":"<<depth_error<<",\"rgb_sum_relative_error\":"<<sum_relative<<",\"alpha_sum_relative_error\":"<<alpha_relative<<"}";
}
inline void photoGrad(Step& s,torch::Tensor loss,std::shared_ptr<GaussianModel> pc) {
    if(!s.active)return;
    std::vector<torch::Tensor> params;for(auto& group:pc->sparse_optimizer_->param_groups())params.push_back(group.params()[0]);
    auto grads=torch::autograd::grad({loss},params,{},true,false,false);
    for(auto& g:grads)s.photo.push_back(g.detach().index_select(0,s.ids).flatten(1).norm(2,1).cpu());
}
inline void before(Step& s,std::shared_ptr<GaussianModel> pc) {
    if(!s.active)return;
    torch::NoGradGuard guard;
    for(auto& group:pc->sparse_optimizer_->param_groups()) {auto p=group.params()[0];s.before.push_back(p.detach().index_select(0,s.ids).clone());s.total.push_back(p.grad().detach().index_select(0,s.ids).flatten(1).norm(2,1).cpu());}
}
inline void after(Step& s,const std::shared_ptr<Dataset>& d,std::shared_ptr<GaussianModel> pc,int view,int step,torch::Tensor visible) {
    if(!s.active)return;
    torch::NoGradGuard guard;std::vector<torch::Tensor> updates;
    int i=0;for(auto& group:pc->sparse_optimizer_->param_groups())updates.push_back((group.params()[0].detach().index_select(0,s.ids)-s.before[i++]).flatten(1).norm(2,1).cpu());
    auto rows=torch::cat({s.labels,s.shape,s.raster.index_select(1,s.ids).t().cpu(),visible.index_select(0,s.ids).to(torch::kFloat32).cpu().unsqueeze(1),torch::stack(s.photo,1),torch::stack(s.total,1),torch::stack(updates,1)},1);
    const auto dir=daeo::folder(d)+"/cohort_trace";std::filesystem::create_directories(dir);
    daeo::binary(dir+"/"+std::to_string(d->all_frame_num_-1)+"_"+std::to_string(view)+"_"+std::to_string(step)+".f32",rows);
}
}
#endif
