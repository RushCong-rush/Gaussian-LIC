#pragma once
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <vector>

// Stable diagnostic IDs survive optional free-space deletion.
namespace daeo {
inline bool enabled() { return std::getenv("ODGS_DAEO_DIAG") != nullptr; }
inline bool tracking() {
    const char* mode=std::getenv("ODGS_DAEO_WARMUP");
    return enabled() || (mode && std::string(mode)!="off");
}
inline bool denseSelected(int f) {
    const char* range=std::getenv("ODGS_DAEO_DENSE_RANGE");
    if(!range) return false;
    const std::string s(range);const auto sep=s.find(':');
    return f>=std::stoi(s.substr(0,sep)) && f<=std::stoi(s.substr(sep+1)) && (f+1)%5==0;
}
inline bool selected(int f) { return denseSelected(f) || (f >= 599 && ((f+1)%50 == 0 || f==934 || f==1034)); }
inline bool auditSelected(int f) {const char* s=std::getenv("ODGS_DAEO_AUDIT_FRAME");return enabled() && s && f==std::stoi(s);}
inline std::vector<float>& labels() { static std::vector<float> v; return v; }
inline int64_t& nextId() { static int64_t n=0; return n; }
inline void prune(torch::Tensor keep) {
    if(!tracking()) return;
    auto k=keep.cpu().contiguous();auto* ptr=k.data_ptr<bool>();auto old=labels();
    TORCH_CHECK((int64_t)old.size()==k.numel()*4,"DAEO prune metadata mismatch");
    labels().clear();
    for(int64_t i=0;i<k.numel();++i) if(ptr[i])
        labels().insert(labels().end(),old.begin()+4*i,old.begin()+4*i+4);
}
inline std::string folder(const std::shared_ptr<Dataset>& d) {
    auto s=d->diagnosis_dir_+"/daeo"; std::filesystem::create_directories(s); return s;
}
inline void binary(const std::string& path, torch::Tensor t) {
    t=t.detach().cpu().to(torch::kFloat32).contiguous();
    std::ofstream f(path,std::ios::binary); f.write((char*)t.data_ptr<float>(), t.numel()*4);
}
inline void append(const std::shared_ptr<Dataset>& d, std::shared_ptr<GaussianModel> pc,
                   torch::Tensor xyz, torch::Tensor scales, torch::Tensor rot,
                   torch::Tensor opacity, torch::Tensor source, torch::Tensor rescued) {
    if(!tracking()) return;
    torch::NoGradGuard g;
    int64_t old=pc->getXYZ().size(0),n=xyz.size(0);
    auto& l=labels(); TORCH_CHECK((int64_t)l.size()==old*4,"DAEO append-only ID mismatch");
    auto s=source.to(torch::kCPU).to(torch::kInt32).contiguous();
    auto a=rescued.to(torch::kCPU).to(torch::kInt32).contiguous();
    int frame=d->all_frame_num_-1;
    for(int64_t i=0;i<n;++i) {l.push_back(nextId()++);l.push_back(frame);l.push_back(s.data_ptr<int>()[i]);l.push_back(a.data_ptr<int>()[i]);}
    if(!enabled()) return;
    auto lab=torch::from_blob(l.data()+old*4,{n,4},torch::kFloat32).clone();
    binary(folder(d)+"/birth_"+std::to_string(frame)+".f32",
        torch::cat({lab,xyz.detach().cpu(),scales.detach().exp().cpu(),
                    torch::nn::functional::normalize(rot.detach()).cpu(),opacity.detach().sigmoid().cpu()},1));
}
inline void initialize(const std::shared_ptr<Dataset>& d,std::shared_ptr<GaussianModel> pc) {
    if(!tracking() || !labels().empty()) return;
    for(int64_t i=0;i<pc->getXYZ().size(0);++i) {auto& l=labels();l.push_back(nextId()++);l.push_back(d->all_frame_num_-1);l.push_back(-1);l.push_back(0);}
}
inline torch::Tensor metadata(std::shared_ptr<GaussianModel> pc) {
    TORCH_CHECK((int64_t)labels().size()==pc->getXYZ().size(0)*4,"DAEO ID mismatch");
    return torch::from_blob(labels().data(),{pc->getXYZ().size(0),4},torch::kFloat32).clone();
}
inline void snapshot(const std::shared_ptr<Dataset>& d,std::shared_ptr<GaussianModel> pc,const std::string& phase, std::shared_ptr<Camera> camera=nullptr) {
    auto c=camera?camera:d->train_cameras_.back();
    int frame=c->frame_index_;
    if(!enabled() || !selected(frame)) return;
    torch::NoGradGuard g;
    auto lab=metadata(pc);auto ids=torch::nonzero(lab.select(1,1)>=frame-70).squeeze(1);
    auto gpu=ids.to(torch::kCUDA);
    binary(folder(d)+"/"+phase+"_"+std::to_string(frame)+".f32",torch::cat({lab.index_select(0,ids),
        pc->getXYZ().index_select(0,gpu).cpu(),pc->getScaling().index_select(0,gpu).cpu(),
        pc->getRotation().index_select(0,gpu).cpu(),pc->getOpacity().index_select(0,gpu).cpu()},1));
    torch::Tensor bg=torch::zeros({3},torch::kFloat32).cuda(),w;
    auto pkg=render(c,pc,bg,false,false,1.f,true,false,&w);
    binary(folder(d)+"/"+phase+"_"+std::to_string(frame)+"_image.f32",torch::cat({std::get<0>(pkg),
       c->original_image_.to(torch::kCUDA),std::get<1>(pkg).reshape({1,c->image_height_,c->image_width_}),w,
       c->diagnostic_depth_.to(torch::kCUDA).reshape({1,c->image_height_,c->image_width_}),
       c->lidar_valid_mask_.to(torch::kCUDA).reshape({1,c->image_height_,c->image_width_})},0));
    std::ofstream f(folder(d)+"/pose_"+std::to_string(frame)+".txt");
    f<<std::setprecision(17)<<c->R_cw_<<'\n'<<c->t_cw_.transpose()<<'\n';
}
struct Step { std::vector<torch::Tensor> ids,before,grad;std::vector<int64_t> counts; };
inline Step beforeStep(const std::shared_ptr<Dataset>& d,std::shared_ptr<GaussianModel> pc,torch::Tensor visible) {
    torch::NoGradGuard guard; Step out;auto lab=metadata(pc).to(torch::kCUDA);auto age=(d->all_frame_num_-1-lab.select(1,1))/5;
    for(int cohort=0;cohort<3;++cohort) {
        auto mask=visible & (cohort==0?age<5:cohort==1?((age>=5)&(age<15)):age>=15);
        auto ids=torch::nonzero(mask).squeeze(1);out.counts.push_back(ids.numel());
        if(ids.numel()>4096) ids=ids.index({torch::indexing::Slice(0,torch::indexing::None,(ids.numel()+4095)/4096)});
        out.ids.push_back(ids);
        for(auto& group:pc->sparse_optimizer_->param_groups()) {auto p=group.params()[0];out.before.push_back(p.detach().index_select(0,ids).clone());out.grad.push_back(p.grad().detach().index_select(0,ids).clone());}
    } return out;
}
inline void afterStep(const std::shared_ptr<Dataset>& d,std::shared_ptr<GaussianModel> pc,Step& st,int view,int step) {
    torch::NoGradGuard guard;auto path=folder(d)+"/updates.csv";bool head=!std::filesystem::exists(path);std::ofstream f(path,std::ios::app);
    if(head) f<<"frame,view,step,cohort,param,visible_count,samples,grad_mean,grad_p90,grad_nonzero,update_mean,update_p90\n";
    const char* names[]={"xyz","dc","sh","opacity_logit","log_scale","rotation"};
    for(int c=0;c<3;++c) {auto ids=st.ids[c];if(!ids.numel())continue;
        for(int p=0;p<6;++p) {auto cur=pc->sparse_optimizer_->param_groups()[p].params()[0].detach().index_select(0,ids);
            auto grad=st.grad[c*6+p].flatten(1).norm(2,1);auto update=(cur-st.before[c*6+p]).flatten(1).norm(2,1);
            f<<d->all_frame_num_-1<<','<<view<<','<<step<<','<<c<<','<<names[p]<<','<<st.counts[c]<<','<<ids.numel()<<','
             <<grad.mean().item<float>()<<','<<torch::quantile(grad,.9).item<float>()<<','<<grad.gt(0).to(torch::kFloat32).mean().item<float>()<<','
             <<update.mean().item<float>()<<','<<torch::quantile(update,.9).item<float>()<<'\n';
        }
    }
}
}
