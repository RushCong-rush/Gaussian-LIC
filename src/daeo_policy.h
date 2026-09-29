#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace daeo {
inline void reallocate(std::vector<int>& views, int train_count, double q) {
    if (!(q>=0 && q<=1) || train_count<1) throw std::invalid_argument("Invalid recent budget");
    int k=std::min(15,train_count),n=static_cast<int>(std::floor(q*views.size()));
    // Preserve total step count and the existing RNG sequence.
    for(int i=0;i<n;++i) views[i]=train_count-1-(i%k);
}
}
