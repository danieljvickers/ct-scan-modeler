// incompressible flow past STL immersed boundaries: one MPI rank per GPU, slab decomposed
#include <cstdio>

#include "out.hpp"

namespace {

// ranks on the same node share its GPUs round robin
int selectDevice() {
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpiRank(), MPI_INFO_NULL, &node_comm);
    int local_rank, num_devices;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_free(&node_comm);
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CTM_REQUIRE(num_devices > 0, "no CUDA device found");
    const int device = local_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device));
    return device;
}

void formatDuration(real seconds, char* text, size_t size) {
    const long whole = static_cast<long>(seconds);
    std::snprintf(text, size, "%ld:%02ld:%02ld", whole/3600, (whole/60) % 60, whole % 60);
}

void run(const char* input_path) {
    const int device = selectDevice();
    const bool is_root = mpiRank() == 0;
    const Parameters parameters = readParameters(input_path);
    checkCaseOptimization(parameters);
    const GridView grid = makeGrid(parameters);
    const long global_cells = static_cast<long>(grid.global_cells[0])*grid.global_cells[1]*grid.global_cells[2];
    if (is_root) {
        std::printf("INFO :: %dD, %d x %d x %d cells on %d GPU(s), %s build\n", grid.num_dims, grid.global_cells[0], grid.global_cells[1],
                    grid.global_cells[2], mpiSize(),
#ifdef CTM_CASE_OPTIMIZATION
                    "case optimized"
#else
                    "generic"
#endif
        );
    }
    std::printf("INFO :: rank %d on GPU %d owns %d cells from %d along axis %d\n", mpiRank(), device, grid.cells[grid.decomposed_axis],
                grid.offset[grid.decomposed_axis], grid.decomposed_axis);

    // set up the bodies, the solver, and the output
    real setup_start = MPI_Wtime();
    ImmersedBoundaries immersed_boundaries(parameters, grid);
    if (is_root && parameters.ib)
        std::printf("INFO :: %ld ghost points, %ld bilinear, %ld from their own neighbors, the rest inverse distance; %ld sealed fluid cells filled\n",
                    immersed_boundaries.geometry.global_ghost_points, immersed_boundaries.geometry.global_bilinear,
                    immersed_boundaries.geometry.global_fallback, immersed_boundaries.geometry.global_trapped);
    if (is_root && parameters.ib && parameters.ib_fill_faces > 0)
        std::printf("INFO :: %ld fluid cells with at least %d solid face neighbors filled\n", immersed_boundaries.geometry.global_thin,
                    parameters.ib_fill_faces);
    FlowSolver solver(parameters, grid, immersed_boundaries);
    Output output(parameters, grid, immersed_boundaries);
    solver.initialize();
    output.writeFields(parameters.t_step_start, parameters.t_step_start*parameters.dt, solver);
    if (is_root) std::printf("INFO :: set up in %.1f s\n", MPI_Wtime() - setup_start);

    // march in time
    const int num_steps = parameters.t_step_stop - parameters.t_step_start;
    const real loop_start = MPI_Wtime();
    bool triggered_divergence = false, triggered_cfl = false;
    for (int step = parameters.t_step_start + 1; step <= parameters.t_step_stop; ++step) {
        const real time = step*parameters.dt;
        solver.advance();
        if (parameters.ib && parameters.ib_force_wrt) output.recordForces(step, time, solver.forces());

        StepDiagnostics diagnostics;
        bool wrote_this_step = false;
        if (parameters.diag_wrt) {
            diagnostics = solver.diagnostics(parameters.dt);
            output.recordDiagnostics(step, time, parameters.dt, diagnostics);
            // save the fields the first time the divergence or CFL number leaves its normal range, so the cells can be inspected
            const bool diverging = !triggered_divergence && diagnostics.divergence.value > 1.0e-6;
            const bool fast = !triggered_cfl && diagnostics.cfl.value > 1.0;
            if (diverging || fast) {
                if (is_root)
                    std::printf("INFO :: step %d t = %.6f s: %s, peak speed %.4f at cell (%d, %d, %d) type %d with %d solid face neighbors, "
                                "CFL %.3f at (%d, %d, %d), divergence %.2e at (%d, %d, %d)\n", step, time,
                                diverging ? "divergence above 1e-6" : "CFL above 1", diagnostics.speed.value, diagnostics.speed.index[0],
                                diagnostics.speed.index[1], diagnostics.speed.index[2], diagnostics.speed.cell_type,
                                diagnostics.speed.solid_neighbors, diagnostics.cfl.value, diagnostics.cfl.index[0], diagnostics.cfl.index[1],
                                diagnostics.cfl.index[2], diagnostics.divergence.value, diagnostics.divergence.index[0],
                                diagnostics.divergence.index[1], diagnostics.divergence.index[2]);
                output.writeFields(step, time, solver);
                wrote_this_step = true;
                triggered_divergence |= diverging;
                triggered_cfl |= fast;
            }
        }

        const bool is_save = step % parameters.t_step_save == 0 || step == parameters.t_step_stop;
        if (parameters.run_time_info || is_save) {
            const real max_divergence = solver.maxDivergence();
            const real elapsed = MPI_Wtime() - loop_start;
            const int steps_done = step - parameters.t_step_start;
            char eta[32];
            formatDuration(elapsed/steps_done*(num_steps - steps_done), eta, sizeof(eta));
            const auto iterations = solver.lastIterations();
            if (is_root)
            {
                std::printf("step %d/%d (%.1f%%) :: max |∇·u| %.2e :: u/v/w/p iterations %d/%d/%d/%d :: %.4f s/step :: ETA %s", step,
                            parameters.t_step_stop, 100.0*steps_done/num_steps, max_divergence, iterations[0], iterations[1], iterations[2],
                            iterations[3], elapsed/steps_done, eta);
                if (parameters.diag_wrt)
                    std::printf(" :: max |u| %.3f at (%d,%d,%d) type %d :: CFL %.2f", diagnostics.speed.value, diagnostics.speed.index[0],
                                diagnostics.speed.index[1], diagnostics.speed.index[2], diagnostics.speed.cell_type, diagnostics.cfl.value);
                std::printf("\n");
            }
        }
        if (!is_save) continue;

        // a diverged solution is saved first, so the last state renders next to every earlier save
        if (!solver.isFinite()) {
            if (!wrote_this_step) output.writeFields(step, time, solver);
            output.flushForces();
            if (is_root) std::fprintf(stderr, "ERROR :: the solution is not finite at step %d, saved it to D/%s and aborting\n", step, "fields.xmf");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (!wrote_this_step) output.writeFields(step, time, solver);
    }

    // timing summary, grind time is the wall time per cell per step
    const real wall = MPI_Wtime() - loop_start;
    if (is_root && num_steps > 0) {
        char total[32];
        formatDuration(wall, total, sizeof(total));
        std::printf("INFO :: %d steps in %s, %.4f s/step, grind time %.2f ns per cell per step\n", num_steps, total, wall/num_steps,
                    1.0e9*wall*mpiSize()/(static_cast<real>(num_steps)*global_cells));
        std::printf("INFO :: open %s/D/fields.xmf in ParaView or VisIt\n", parameters.case_dir.c_str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CTM_REQUIRE(argc == 2, "usage: simulation <input file written by ./ctm.sh run>");
    run(argv[1]);  // every GPU resource is released when run returns, before MPI_Finalize
    MPI_Finalize();
    return EXIT_SUCCESS;
}
