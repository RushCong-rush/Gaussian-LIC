#pragma once
// LiDAR free-space deletion. No splitting, cloning or generic opacity pruning.
#ifdef ODGS_FREE_SPACE_FOOTPRINT
extern "C" void freeSpaceFootprintPointers(const float*,const unsigned char*,float*,int,float,float);
#endif
namespace lidar_free_space {
inline float option(const char* name,float fallback) {
    const char* s=std::getenv(name);return s?std::stof(s):fallback;
}
inline void erase(std::shared_ptr<GaussianModel> pc,torch::Tensor keep) {
    auto& state=pc->sparse_optimizer_->get_state();
    std::vector<torch::Tensor> result;
    for(auto& group:pc->sparse_optimizer_->param_groups()) {
        auto& p=group.params()[0];auto old=p.unsafeGetTensorImpl();
        auto q=p.detach().index({keep}).contiguous().requires_grad_();
        auto it=state.find(old);
        if(it!=state.end()) {
            auto saved=it->second;state.erase(it);
            saved.exp_avg=saved.exp_avg.index({keep}).contiguous();
            saved.exp_avg_sq=saved.exp_avg_sq.index({keep}).contiguous();
            state[q.unsafeGetTensorImpl()]=saved;
        }
        p=q;result.push_back(q);
    }
    TORCH_CHECK(result.size()==6,"Expected six Gaussian parameter groups");
    pc->xyz_=result[0];pc->features_dc_=result[1];pc->features_rest_=result[2];
    pc->opacity_=result[3];pc->scaling_=result[4];pc->rotation_=result[5];
    pc->Tensor_vec_xyz_={result[0]};pc->Tensor_vec_feature_dc_={result[1]};
    pc->Tensor_vec_feature_rest_={result[2]};pc->Tensor_vec_opacity_={result[3]};
    pc->Tensor_vec_scaling_={result[4]};pc->Tensor_vec_rotation_={result[5]};
    daeo::prune(keep);
}
inline void apply(const std::shared_ptr<Dataset>& d,std::shared_ptr<GaussianModel> pc) {
    if(!d->equirectangular_ || option("ODGS_FREE_SPACE_PRUNE",1)==0) return;
    torch::NoGradGuard guard;auto start=std::chrono::steady_clock::now();
    auto c=d->train_cameras_.back();int H=c->image_height_,W=c->image_width_;
    // Only measured LiDAR pixels; no dense DAP depths are used as free-space evidence.
    auto dep=c->diagnostic_depth_.to(torch::kCUDA).reshape({H,W});
    auto valid=c->lidar_valid_mask_.to(torch::kCUDA).reshape({H,W}).to(torch::kBool)&torch::isfinite(dep)&(dep>0);
    if(d->metric_mask_.defined()) valid=valid&d->metric_mask_.to(torch::kCUDA).reshape({H,W}).to(torch::kBool);
    auto lo=torch::full_like(dep,1e8),hi=torch::zeros_like(dep),count=torch::zeros_like(dep);
    // A periodic 5x5 neighborhood uses coherent measured returns only.
    for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx) {
        auto z=torch::roll(dep,{dy,dx},{0,1});auto m=torch::roll(valid,{dy,dx},{0,1});
        if(dy>0)m.index_put_({torch::indexing::Slice(0,dy)},false);
        if(dy<0)m.index_put_({torch::indexing::Slice(H+dy,H)},false);
        lo=torch::minimum(lo,torch::where(m,z,torch::full_like(z,1e8)));
        hi=torch::maximum(hi,torch::where(m,z,torch::zeros_like(z)));
        count+=m.to(torch::kFloat32);
    }
    auto tol=torch::maximum(torch::full_like(lo,pc->map_extension_min_depth_gap_m_),lo*pc->map_extension_relative_depth_gap_);
    const int min_support=int(option("ODGS_FREE_SPACE_MIN_SUPPORT",1));
    auto reliable=(count>=min_support)&((hi-lo)<=tol);
    auto original_reliable=reliable.clone();
    // Optional small-footprint support. Copies never count as independent returns.
    const int support_patch=int(option("ODGS_FREE_SPACE_SUPPORT_PATCH",0));
    if(support_patch) {
        TORCH_CHECK(support_patch==2 || support_patch==3,"Support patch must be 0, 2 or 3");
        auto supported=torch::zeros_like(valid);
        const int first=support_patch==2?0:-1;
        for(int dy=first;dy<=1;++dy)for(int dx=first;dx<=1;++dx) {
            auto m=torch::roll(valid,{dy,dx},{0,1});
            if(dy>0)m.index_put_({torch::indexing::Slice(0,dy)},false);
            if(dy<0)m.index_put_({torch::indexing::Slice(H+dy,H)},false);
            supported|=m;
        }
        // Keep the original 5x5 minimum depth and consistency check at boundaries.
        reliable|=supported&((hi-lo)<=tol);
        if(d->metric_mask_.defined())reliable&=d->metric_mask_.to(torch::kCUDA).reshape({H,W}).to(torch::kBool);
    }
    auto xyz=pc->getXYZ().detach();int64_t n=xyz.size(0);
    auto hom=torch::cat({xyz,torch::ones({n,1},xyz.options())},1);
    auto cam=hom.matmul(c->world_view_transform_).index({torch::indexing::Slice(),torch::indexing::Slice(0,3)});
    auto x=cam.select(1,0),y=cam.select(1,1),z=cam.select(1,2);auto r=cam.norm(2,1);
    auto u=torch::remainder((torch::atan2(x,z)/(2*M_PI)+.5)*W,W).to(torch::kLong);
    auto v=((torch::atan2(y,torch::sqrt(x*x+z*z))/M_PI+.5)*H).to(torch::kLong).clamp(0,H-1);
    auto pix=v*W+u;auto ref=lo.flatten().index_select(0,pix);
    auto gap=tol.flatten().index_select(0,pix);
    auto sigma=std::get<0>(pc->getScaling().detach().max(1));auto op=pc->getOpacity().detach().flatten();
    auto large=(sigma>=option("ODGS_FREE_SPACE_SIGMA",.1f))&(sigma>=r*option("ODGS_FREE_SPACE_ANGULAR",.1f));
    auto conflict=reliable.flatten().index_select(0,pix)&(r+gap<ref)&(op>=option("ODGS_FREE_SPACE_OPACITY",.5f))&large;
    if(pc->skybox_points_num_>0)conflict.index_put_({torch::indexing::Slice(0,pc->skybox_points_num_)},false);
    auto footprint_conflict=torch::zeros_like(conflict);
    torch::Tensor footprint_weights;
#ifdef ODGS_FREE_SPACE_FOOTPRINT
    if(option("ODGS_FREE_SPACE_FOOTPRINT",1)!=0) {
        auto candidates=((count.flatten().index_select(0,pix)==0)&large&
            (op>=option("ODGS_FREE_SPACE_OPACITY",.5f))).to(torch::kUInt8).contiguous();
        if(pc->skybox_points_num_>0)candidates.index_put_({torch::indexing::Slice(0,pc->skybox_points_num_)},0);
        auto references=torch::where(valid&((hi-lo)<=tol),lo,torch::zeros_like(lo)).contiguous();
        footprint_weights=torch::zeros({3,n},xyz.options());footprint_weights.select(0,2).fill_(1e8);
        auto bg=torch::zeros({3},xyz.options());
        freeSpaceFootprintPointers(references.data_ptr<float>(),candidates.data_ptr<uint8_t>(),footprint_weights.data_ptr<float>(),n,pc->map_extension_min_depth_gap_m_,pc->map_extension_relative_depth_gap_);
        auto check_render=render(c,pc,bg,false,false,1.f,true,false);
        torch::cuda::synchronize();freeSpaceFootprintPointers(nullptr,nullptr,nullptr,0,0,0);
        auto front=footprint_weights.select(0,0),total=footprint_weights.select(0,1);
        footprint_conflict=(candidates>0)&(front>=option("ODGS_FREE_SPACE_FOOTPRINT_WEIGHT",1.f))&
            (front>=option("ODGS_FREE_SPACE_FOOTPRINT_RATIO",.8f)*total);
        conflict|=footprint_conflict;
        ref=torch::where(footprint_conflict,footprint_weights.select(0,2),ref);
    }
#endif
    static torch::Tensor votes;
    if(!votes.defined())votes=torch::zeros({n},xyz.options());
    TORCH_CHECK(votes.numel()<=n,"Free-space vote alignment failed");
    if(votes.numel()<n)votes=torch::cat({votes,torch::zeros({n-votes.numel()},xyz.options())});
    votes=torch::where(conflict,votes+1,torch::zeros_like(votes));
    auto lidar_remove=votes>=option("ODGS_FREE_SPACE_VOTES",1);
    auto dap_remove=torch::zeros_like(lidar_remove),dap_conflict=torch::zeros_like(conflict);
    static torch::Tensor dap_votes;
    auto deletion_ref=ref;
    if(option("ODGS_FREE_SPACE_DAP",0)!=0 && c->has_dap_depth_) {
        // DAP is weaker evidence: no nearby LiDAR, coherent depth and a larger gap.
        auto dense_valid=torch::isfinite(dep)&(dep>0);
        if(d->metric_mask_.defined())dense_valid=dense_valid&d->metric_mask_.to(torch::kCUDA).reshape({H,W}).to(torch::kBool);
        auto dlo=torch::full_like(dep,1e8),dhi=torch::zeros_like(dep),dcnt=torch::zeros_like(dep);
        for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
            auto zz=torch::roll(dep,{dy,dx},{0,1});auto mm=torch::roll(dense_valid,{dy,dx},{0,1});
            if(dy>0)mm.index_put_({torch::indexing::Slice(0,dy)},false);
            if(dy<0)mm.index_put_({torch::indexing::Slice(H+dy,H)},false);
            dlo=torch::minimum(dlo,torch::where(mm,zz,torch::full_like(zz,1e8)));
            dhi=torch::maximum(dhi,torch::where(mm,zz,torch::zeros_like(zz)));dcnt+=mm.to(torch::kFloat32);
        }
        auto dtol=torch::maximum(torch::full_like(dlo,.25),dlo*.1);
        auto coherent=(dcnt==25)&(count==0)&((dhi-dlo)<=dtol);
        auto dref=dlo.flatten().index_select(0,pix);
        auto dgap=torch::maximum(torch::full_like(dref,option("ODGS_FREE_SPACE_DAP_GAP",1.f)),dref*option("ODGS_FREE_SPACE_DAP_REL",.3f));
        dap_conflict=coherent.flatten().index_select(0,pix)&(r+dgap<dref)&large&(op>=option("ODGS_FREE_SPACE_OPACITY",.5f));
        if(pc->skybox_points_num_>0)dap_conflict.index_put_({torch::indexing::Slice(0,pc->skybox_points_num_)},false);
        if(!dap_votes.defined())dap_votes=torch::zeros({n},xyz.options());
        if(dap_votes.numel()<n)dap_votes=torch::cat({dap_votes,torch::zeros({n-dap_votes.numel()},xyz.options())});
        dap_votes=torch::where(dap_conflict,dap_votes+1,torch::zeros_like(dap_votes));
        dap_remove=dap_votes>=option("ODGS_FREE_SPACE_DAP_VOTES",3);
        deletion_ref=torch::where(dap_remove,dref,ref);
    }
    auto remove=lidar_remove|dap_remove;int64_t nr=remove.sum().item<int64_t>();
    auto folder=d->diagnosis_dir_+"/free_space";
    if(!d->diagnosis_dir_.empty()) std::filesystem::create_directories(folder);
    if(nr) {
        if(daeo::enabled()) {
        auto ids=torch::nonzero(remove).squeeze(1);
        auto labels=daeo::enabled()?daeo::metadata(pc).to(torch::kCUDA):torch::zeros({n,4},xyz.options());
        daeo::binary(folder+"/removed_"+std::to_string(c->frame_index_)+".f32",torch::cat({labels.index_select(0,ids),xyz.index_select(0,ids),pc->getScaling().index_select(0,ids),pc->getRotation().index_select(0,ids),pc->getOpacity().index_select(0,ids),deletion_ref.index_select(0,ids).unsqueeze(1),r.index_select(0,ids).unsqueeze(1)},1));
        auto evidence=dap_remove.index_select(0,ids).to(torch::kCPU).to(torch::kUInt8).contiguous();
        std::ofstream ef(folder+"/removed_evidence_"+std::to_string(c->frame_index_)+".u8",std::ios::binary);ef.write((char*)evidence.data_ptr<uint8_t>(),evidence.numel());
        if(footprint_weights.defined()) {
            auto fp=(footprint_conflict&lidar_remove).index_select(0,ids).to(torch::kCPU).to(torch::kUInt8).contiguous();
            std::ofstream ff(folder+"/removed_footprint_"+std::to_string(c->frame_index_)+".u8",std::ios::binary);ff.write((char*)fp.data_ptr<uint8_t>(),fp.numel());
            daeo::binary(folder+"/footprint_weights_"+std::to_string(c->frame_index_)+".f32",footprint_weights.index_select(1,ids).transpose(0,1));
        }
        }
        auto keep=~remove;erase(pc,keep);votes=votes.index({keep});
        if(dap_votes.defined())dap_votes=dap_votes.index({keep});
    }
    if(d->diagnosis_dir_.empty()) return;
    auto path=folder+"/events.csv";bool head=!std::filesystem::exists(path);std::ofstream f(path,std::ios::app);
    if(head)f<<"frame,gs_before,reliable_pixels,conflict_gs,removed_gs,elapsed_ms,lidar_removed_gs,dap_removed_gs,dap_conflict_gs,extra_support_pixels,extra_support_removed_gs,footprint_removed_gs\n";
    torch::cuda::synchronize();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    f<<c->frame_index_<<','<<n<<','<<reliable.sum().item<int64_t>()<<','<<conflict.sum().item<int64_t>()<<','<<nr<<','<<ms<<','<<lidar_remove.sum().item<int64_t>()<<','<<dap_remove.sum().item<int64_t>()<<','<<dap_conflict.sum().item<int64_t>()<<','<<(reliable&~original_reliable).sum().item<int64_t>()<<','<<(lidar_remove&~original_reliable.flatten().index_select(0,pix)&~footprint_conflict).sum().item<int64_t>()<<','<<(lidar_remove&footprint_conflict).sum().item<int64_t>()<<'\n';
}
}
