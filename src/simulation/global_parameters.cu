#include "global_parameters.hpp"

#include <fstream>
#include <map>
#include <sstream>

namespace {

class InputFile {
public:
    explicit InputFile(const std::string& path) {
        std::ifstream input(path);
        CTM_REQUIRE(input.good(), "could not open input file %s", path.c_str());
        std::string line;
        while (std::getline(input, line)) {
            const auto split = line.find(' ');
            if (split == std::string::npos) continue;
            this->values[line.substr(0, split)] = line.substr(split + 1);
        }
    }

    const std::string& text(const std::string& key) const {
        const auto entry = this->values.find(key);
        CTM_REQUIRE(entry != this->values.end(), "input file is missing %s, rerun through ./ctm.sh", key.c_str());
        return entry->second;
    }
    real number(const std::string& key) const { return std::stod(this->text(key)); }
    int integer(const std::string& key) const { return std::stoi(this->text(key)); }
    bool logical(const std::string& key) const { return this->text(key) == "T"; }

private:
    std::map<std::string, std::string> values;
};

const char* direction_names[max_dims] = {"x", "y", "z"};

}  // namespace

Parameters readParameters(const std::string& path) {
    const InputFile input(path);
    Parameters parameters;
    parameters.case_dir = input.text("case_dir");
    parameters.run_time_info = input.logical("run_time_info");

    // computational domain
    parameters.global_cells[0] = input.integer("m") + 1;
    parameters.global_cells[1] = input.integer("n") + 1;
    parameters.global_cells[2] = input.integer("p") + 1;
    parameters.num_dims = parameters.global_cells[2] > 1 ? 3 : 2;
    for (int d = 0; d < max_dims; ++d) {
        const std::string name = direction_names[d];
        parameters.domain_begin[d] = input.number(name + "_domain%beg");
        parameters.domain_end[d] = input.number(name + "_domain%end");
        CTM_REQUIRE(!input.logical("stretch_" + name), "the CUDA solver needs a uniform grid, turn off stretch_%s", name.c_str());
        parameters.boundary[d][0] = input.integer("bc_" + name + "%beg");
        parameters.boundary[d][1] = input.integer("bc_" + name + "%end");
        CTM_REQUIRE((parameters.boundary[d][0] == bc_periodic) == (parameters.boundary[d][1] == bc_periodic),
                    "bc_%s%%beg and bc_%s%%end must both be periodic or neither", name.c_str(), name.c_str());
        for (int component = 0; component < max_dims; ++component)
            parameters.inflow_velocity[d][component] = input.number("bc_" + name + "%vel_in(" + std::to_string(component + 1) + ")");
    }
    if (parameters.num_dims == 2) parameters.domain_end[2] = parameters.domain_begin[2] + 1.0;  // unit depth keeps volumes finite

    // time stepping and solvers
    parameters.dt = input.number("dt");
    parameters.t_step_start = input.integer("t_step_start");
    parameters.t_step_stop = input.integer("t_step_stop");
    parameters.t_step_save = input.integer("t_step_save");
    parameters.viscous = input.logical("viscous");
    parameters.kinematic_viscosity = parameters.viscous ? 1.0/input.number("fluid_pp(1)%Re(1)") : 0.0;
    CTM_REQUIRE(input.integer("poisson_solver") == 2, "the CUDA solver needs poisson_solver = 2");
    parameters.krylov_tolerance = input.number("krylov_tolerance");
    parameters.krylov_restart = input.integer("krylov_restart");
    parameters.krylov_max_iterations = input.integer("krylov_max_iterations");

    // initial condition patches
    for (int patch_id = 1; patch_id <= input.integer("num_patches"); ++patch_id) {
        const std::string prefix = "patch_icpp(" + std::to_string(patch_id) + ")%";
        InitialPatch patch;
        patch.geometry = input.integer(prefix + "geometry");
        patch.hcid = input.integer(prefix + "hcid");
        patch.radius = input.number(prefix + "radius");
        patch.vortex_velocity = input.number(prefix + "vortex_velocity");
        CTM_REQUIRE(patch.geometry == patch_rectangle || (patch.geometry == patch_hardcoded_2d && patch.hcid == hcid_gaussian_vortex),
                    "patch_icpp(%d) must be a rectangle or the hardcoded Gaussian vortex", patch_id);
        for (int d = 0; d < max_dims; ++d) {
            const std::string name = direction_names[d];
            patch.centroid[d] = input.number(prefix + name + "_centroid");
            patch.length[d] = input.number(prefix + "length_" + name);
            patch.velocity[d] = input.number(prefix + "vel(" + std::to_string(d + 1) + ")");
        }
        if (parameters.num_dims == 2) patch.length[2] = 1.0e300;  // ignore z in 2D
        patch.pressure = input.number(prefix + "pres");
        parameters.initial_patches.push_back(patch);
    }

    // immersed boundaries
    parameters.ib = input.logical("ib");
    parameters.gp_layers = input.integer("gp_layers");
    if (parameters.ib) {
        for (int patch_id = 1; patch_id <= input.integer("num_ibs"); ++patch_id) {
            const std::string prefix = "patch_ib(" + std::to_string(patch_id) + ")%";
            ImmersedPatch patch;
            patch.stl_path = input.text(prefix + "geometry");
            patch.threshold = input.number(prefix + "model_threshold");
            for (int d = 0; d < max_dims; ++d) {
                const std::string index = "(" + std::to_string(d + 1) + ")";
                patch.scale[d] = input.number(prefix + "model_scale" + index);
                patch.translate[d] = input.number(prefix + "model_translate" + index);
                patch.rotate[d] = input.number(prefix + "model_rotate" + index);
            }
            parameters.immersed_patches.push_back(patch);
        }
    }

    parameters.prim_vars_wrt = input.logical("prim_vars_wrt");
    parameters.ib_force_wrt = input.logical("ib_force_wrt");
    return parameters;
}

void checkCaseOptimization(const Parameters& parameters) {
#ifdef CTM_CASE_OPTIMIZATION
    const bool matches = case_parameters::m + 1 == parameters.global_cells[0] && case_parameters::n + 1 == parameters.global_cells[1] &&
                         case_parameters::p + 1 == parameters.global_cells[2] && case_parameters::viscous == parameters.viscous &&
                         case_parameters::ib == parameters.ib && case_parameters::gp_layers == parameters.gp_layers &&
                         case_parameters::krylov_restart == parameters.krylov_restart &&
                         (!parameters.ib || case_parameters::num_ibs == static_cast<int>(parameters.immersed_patches.size()));
    CTM_REQUIRE(matches, "this binary was case optimized for different parameters (%s), rebuild through ./ctm.sh run",
                case_parameters::summary);
#else
    (void)parameters;
#endif
}
