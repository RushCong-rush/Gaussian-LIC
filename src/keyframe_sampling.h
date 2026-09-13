#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// Select distinct views within the existing optimization budget.
inline std::vector<int> sampleRecentKeyframes(int count, int budget, int recent,
                                              std::mt19937& generator)
{
    if (count < 0 || budget < 0 || recent < 0)
        throw std::invalid_argument("Keyframe sampling parameters must be nonnegative");
    budget = std::min(budget, count);
    recent = std::min(recent, budget);
    std::vector<int> history(count - recent);
    std::iota(history.begin(), history.end(), 0);
    std::vector<int> selected;
    selected.reserve(budget);
    std::sample(history.begin(), history.end(), std::back_inserter(selected),
                budget - recent, generator);
    for (int index = count - recent; index < count; ++index)
        selected.push_back(index);
    return selected;
}
