// HDF5 field output with an XDMF index that VisIt and ParaView open directly, rewritten after every save
#ifndef CTM_OUT_HPP
#define CTM_OUT_HPP

#include <cstdio>
#include <string>
#include <vector>

#include "right_hand_side.hpp"

class Output {
public:
    Output(const Parameters& parameters, const GridView& grid, const ImmersedBoundaries& immersed_boundaries);
    ~Output();

    // one HDF5 file per save, each rank writes its slab; the XDMF index then lists every save so far
    void writeFields(int step, real time, FlowSolver& solver);

    // forces go to D/ib_forces.dat, buffered on rank 0 and flushed at every save
    void recordForces(int step, real time, const std::vector<real>& forces);
    void flushForces();

    // D/diagnostics.dat, one line per step with the peaks of speed, CFL number, divergence, and pressure and the cells they sit in
    void recordDiagnostics(int step, real time, real dt, const StepDiagnostics& diagnostics);

private:
    void writeGrid(const ImmersedBoundaries& immersed_boundaries);
    void writeXdmf() const;

    const Parameters& parameters;
    GridView grid;
    std::string output_dir;
    std::vector<std::pair<int, real>> saves;  // (step, time)
    std::vector<std::string> force_lines;
    std::FILE* force_file = nullptr;
    std::FILE* diagnostics_file = nullptr;
    DeviceArray<real> compact;
};

#endif
