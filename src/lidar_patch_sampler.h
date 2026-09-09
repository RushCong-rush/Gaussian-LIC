#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// Feed only eligible candidates; preserve their original indices/3D coordinates.
class LidarPatchSampler
{
public:
    LidarPatchSampler(int width, int height, int patch_size)
        : width_(width), height_(height), patch_size_(patch_size)
    {
        if (width <= 0 || height <= 0 || patch_size <= 0)
            throw std::invalid_argument("Patch sampling dimensions must be positive");
        cols_ = (width + patch_size - 1) / patch_size;
        const int rows = (height + patch_size - 1) / patch_size;
        indices_.assign(cols_ * rows, -1);
        depths_.assign(indices_.size(), std::numeric_limits<double>::infinity());
    }

    void consider(int index, int u, int v, double depth)
    {
        if (u < 0 || u >= width_ || v < 0 || v >= height_ ||
            !std::isfinite(depth) || depth <= 0.0)
            return;
        const int patch = (v / patch_size_) * cols_ + u / patch_size_;
        if (depth < depths_[patch] ||
            (depth == depths_[patch] && index < indices_[patch]))
        {
            depths_[patch] = depth;
            indices_[patch] = index;
        }
    }

    const std::vector<int>& indices() const { return indices_; }

private:
    int width_, height_, patch_size_, cols_;
    std::vector<int> indices_;
    std::vector<double> depths_;
};
