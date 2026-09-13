#include "keyframe_sampling.h"
#include <cassert>
#include <iostream>
#include <set>

int main()
{
    for (int count : {0, 1, 9, 10, 50, 99, 100, 101, 216, 1000})
        for (int budget : {0, 5, 50, 100})
            for (int recent : {0, 5, 10, 15, 100})
                for (unsigned seed = 0; seed < 20; ++seed)
                {
                    std::mt19937 generator(seed);
                    const auto views = sampleRecentKeyframes(count, budget, recent, generator);
                    const int size = std::min(count, budget);
                    std::set<int> unique(views.begin(), views.end());
                    assert(static_cast<int>(views.size()) == size);
                    assert(unique.size() == views.size());
                    for (int view : views) assert(view >= 0 && view < count);
                    for (int view = count - std::min(recent, size); view < count; ++view)
                        assert(unique.count(view));
                }
    // Identical seeds are reproducible; older views still participate.
    std::mt19937 a(7), b(7);
    assert(sampleRecentKeyframes(216, 100, 10, a) == sampleRecentKeyframes(216, 100, 10, b));
    std::set<int> history;
    for (unsigned seed = 0; seed < 100; ++seed)
    {
        std::mt19937 g(seed);
        auto views = sampleRecentKeyframes(216, 100, 10, g);
        for (int v : views) if (v < 206) history.insert(v);
    }
    assert(history.size() == 206);
    std::cout << "Keyframe sampling tests passed\n";
}
