#include "mapping.h"

#include <filesystem>

static void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv)
{
    require(argc == 2, "Usage: mapping_params_test CONFIG_DIRECTORY");
    size_t tested = 0;
    for (const auto& file : std::filesystem::directory_iterator(argv[1]))
    {
        if (file.path().extension() != ".yaml") continue;
        const std::string path = file.path().string();
        const YAML::Node original = YAML::LoadFile(path);
        const Params baseline(original);
        require(baseline.lidar_patch_size == 3, path + ": production profile must use LiDAR patch 3");

        require(baseline.optimization_recent_keyframes == 10, path + ": default recent window must be 10");
        for (int recent : {0, 5, 10, 15, 100})
        {
            auto sampling = YAML::Clone(original);
            sampling["optimization_recent_keyframes"] = recent;
            require(Params(sampling).optimization_recent_keyframes == recent, "Sampling override ignored");
        }
        for (int recent : {-1, 101})
        {
            auto sampling = YAML::Clone(original);
            sampling["optimization_recent_keyframes"] = recent;
            bool rejected = false;
            try { Params invalid(sampling); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Invalid recent window accepted");
        }
        auto config = YAML::Clone(original);
        config.remove("lidar_patch_size");
        require(Params(config).lidar_patch_size == 3, path + ": omitted value must default to 3");
        for (int size : {0, 2, 3, 8})
        {
            config["lidar_patch_size"] = size;
            const Params parameters(config);
            require(parameters.lidar_patch_size == size, path + ": explicit override ignored");
            require(parameters.patch_size == baseline.patch_size, path + ": DAP patch changed");
            require(parameters.map_extension_min_depth_gap_m == baseline.map_extension_min_depth_gap_m &&
                    parameters.map_extension_relative_depth_gap == baseline.map_extension_relative_depth_gap,
                    path + ": depth-aware extension changed");
            require(parameters.lambda_depth == baseline.lambda_depth &&
                    parameters.dap_dense_depth_supervision == baseline.dap_dense_depth_supervision,
                    path + ": depth supervision changed");
        }
        config["lidar_patch_size"] = -1;
        bool rejected = false;
        try { Params invalid(config); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, path + ": negative patch size accepted");
        ++tested;
    }
    require(tested > 0, "No production profiles tested");
    std::cout << "Mapping parameter tests passed for " << tested << " profiles\n";
}
