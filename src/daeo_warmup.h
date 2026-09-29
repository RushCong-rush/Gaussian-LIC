#pragma once
// Temporary extra optimization of recently depth-rescued Gaussians.
namespace daeo_warmup {
inline std::string mode() {const char* s=std::getenv("ODGS_DAEO_WARMUP");return s?s:"off";}
inline int steps() {const char* s=std::getenv("ODGS_DAEO_WARMUP_STEPS");const int n=s?std::stoi(s):10;TORCH_CHECK(n>0,"DAEO warmup steps must be positive");return n;}
struct FreezeAudit {
    torch::Tensor ids;
    std::vector<torch::Tensor> params,first,second;
};
inline FreezeAudit before(std::shared_ptr<GaussianModel> pc,torch::Tensor allowed) {
    torch::NoGradGuard guard;FreezeAudit a;
    a.ids=torch::nonzero(~allowed).squeeze(1);
    if(a.ids.numel()>4096)a.ids=a.ids.index({torch::indexing::Slice(0,torch::indexing::None,(a.ids.numel()+4095)/4096)});
    for(auto& group:pc->sparse_optimizer_->param_groups()) {
        auto p=group.params()[0];auto& state=pc->sparse_optimizer_->get_state().at(p.unsafeGetTensorImpl());
        a.params.push_back(p.detach().index_select(0,a.ids).clone());
        a.first.push_back(state.exp_avg.index_select(0,a.ids).clone());
        a.second.push_back(state.exp_avg_sq.index_select(0,a.ids).clone());
    }
    return a;
}
inline void after(const std::shared_ptr<Dataset>& d,std::shared_ptr<GaussianModel> pc,
                  const FreezeAudit& a,int view,int step) {
    if(!a.ids.numel())return;
    torch::NoGradGuard guard;double delta=0,m1=0,m2=0;int k=0;
    for(auto& group:pc->sparse_optimizer_->param_groups()) {
        auto p=group.params()[0];auto& state=pc->sparse_optimizer_->get_state().at(p.unsafeGetTensorImpl());
        delta=std::max(delta,(p.detach().index_select(0,a.ids)-a.params[k]).abs().max().item<double>());
        m1=std::max(m1,(state.exp_avg.index_select(0,a.ids)-a.first[k]).abs().max().item<double>());
        m2=std::max(m2,(state.exp_avg_sq.index_select(0,a.ids)-a.second[k]).abs().max().item<double>());++k;
    }
    TORCH_CHECK(delta==0 && m1==0 && m2==0,"Warmup changed frozen GS parameters or Adam moments");
    auto path=d->diagnosis_dir_+"/warmup_freeze_audit.csv";bool header=!std::filesystem::exists(path);std::ofstream f(path,std::ios::app);
    if(header)f<<"frame,view,step,sampled_frozen_gs,parameter_delta,moment1_delta,moment2_delta\n";
    f<<d->all_frame_num_-1<<','<<view<<','<<step<<','<<a.ids.numel()<<','<<delta<<','<<m1<<','<<m2<<'\n';
}
}
