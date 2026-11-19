#include "out.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include <hdf5.h>

namespace {

#define HDF5_CHECK(call) CTM_REQUIRE((call) >= 0, "HDF5 call %s failed", #call)

template <typename T>
__global__ void gatherOwnedAsRealKernel(GridView grid, const T* storage, real* compact) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    compact[owned] = static_cast<real>(storage[grid.storageIndex(i, j, k)]);
}

std::string stepFilename(int step) {
    char name[64];
    std::snprintf(name, sizeof(name), "fields_%08d.h5", step);
    return name;
}

hid_t createParallelFile(const std::string& path) {
    const hid_t access = H5Pcreate(H5P_FILE_ACCESS);
    HDF5_CHECK(H5Pset_fapl_mpio(access, MPI_COMM_WORLD, MPI_INFO_NULL));
    const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access);
    HDF5_CHECK(file);
    H5Pclose(access);
    return file;
}

// dataset dims are slowest first, (z, y, x) in 3D and (y, x) in 2D, matching the compact layout and XDMF
void writeSlab(hid_t file, const char* name, const GridView& grid, const real* host_compact) {
    const int rank = grid.num_dims;
    hsize_t global_dims[max_dims], local_dims[max_dims], offsets[max_dims];
    for (int r = 0; r < rank; ++r) {
        const int d = rank - 1 - r;
        global_dims[r] = grid.global_cells[d];
        local_dims[r] = grid.cells[d];
        offsets[r] = grid.offset[d];
    }
    const hid_t file_space = H5Screate_simple(rank, global_dims, nullptr);
    const hid_t dataset = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    HDF5_CHECK(dataset);
    HDF5_CHECK(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offsets, nullptr, local_dims, nullptr));
    const hid_t memory_space = H5Screate_simple(rank, local_dims, nullptr);
    const hid_t transfer = H5Pcreate(H5P_DATASET_XFER);
    HDF5_CHECK(H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE));
    HDF5_CHECK(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space, transfer, host_compact));
    H5Pclose(transfer);
    H5Sclose(memory_space);
    H5Dclose(dataset);
    H5Sclose(file_space);
}

// a small array every rank creates collectively, only rank 0 supplies the data
void writeRootArray(hid_t file, const char* name, const std::vector<real>& values) {
    const hsize_t count = values.size();
    const hid_t file_space = H5Screate_simple(1, &count, nullptr);
    const hid_t dataset = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    HDF5_CHECK(dataset);
    const hid_t memory_space = H5Screate_simple(1, &count, nullptr);
    if (mpiRank() != 0) {
        H5Sselect_none(file_space);
        H5Sselect_none(memory_space);
    }
    const hid_t transfer = H5Pcreate(H5P_DATASET_XFER);
    HDF5_CHECK(H5Pset_dxpl_mpio(transfer, H5FD_MPIO_COLLECTIVE));
    HDF5_CHECK(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space, transfer, values.data()));
    H5Pclose(transfer);
    H5Sclose(memory_space);
    H5Dclose(dataset);
    H5Sclose(file_space);
}

}  // namespace

Output::Output(const Parameters& parameters, const GridView& grid, const ImmersedBoundaries& immersed_boundaries)
    : parameters(parameters), grid(grid), output_dir(parameters.case_dir + "/D") {
    if (mpiRank() == 0) mkdir(this->output_dir.c_str(), 0755);
    MPI_Barrier(MPI_COMM_WORLD);
    this->compact.resize(grid.numOwned());
    this->writeGrid(immersed_boundaries);

    if (mpiRank() == 0 && parameters.ib && parameters.ib_force_wrt) {
        this->force_file = std::fopen((this->output_dir + "/ib_forces.dat").c_str(), "w");
        CTM_REQUIRE(this->force_file != nullptr, "could not open %s/ib_forces.dat", this->output_dir.c_str());
        std::fprintf(this->force_file, "# step time");
        for (size_t patch = 1; patch <= parameters.immersed_patches.size(); ++patch)
            std::fprintf(this->force_file, " force_x(%zu) force_y(%zu) force_z(%zu)", patch, patch, patch);
        std::fprintf(this->force_file, "\n");
    }
}

Output::~Output() {
    this->flushForces();
    if (this->force_file) std::fclose(this->force_file);
}

void Output::writeGrid(const ImmersedBoundaries& immersed_boundaries) {
    const hid_t file = createParallelFile(this->output_dir + "/grid.h5");
    const char* node_names[max_dims] = {"x_nodes", "y_nodes", "z_nodes"};
    for (int d = 0; d < this->grid.num_dims; ++d) {
        std::vector<real> nodes(this->grid.global_cells[d] + 1);
        for (size_t node = 0; node < nodes.size(); ++node) nodes[node] = this->grid.origin[d] + node*this->grid.spacing[d];
        writeRootArray(file, node_names[d], nodes);
    }
    gatherOwnedAsRealKernel<<<numBlocks(this->grid.numOwned()), block_size>>>(this->grid, immersed_boundaries.geometry.markers.data(),
                                                                               this->compact.data());
    std::vector<real> host(this->grid.numOwned());
    this->compact.copyToHost(host.data());
    writeSlab(file, "ib_markers", this->grid, host.data());
    H5Fclose(file);
}

void Output::writeFields(int step, real time, FlowSolver& solver) {
    if (!this->parameters.prim_vars_wrt) return;
    const hid_t file = createParallelFile(this->output_dir + "/" + stepFilename(step));
    const char* velocity_names[max_dims] = {"u", "v", "w"};
    std::vector<real> host(this->grid.numOwned());
    auto write = [&](const char* name, const real* storage) {
        gatherOwnedAsRealKernel<<<numBlocks(this->grid.numOwned()), block_size>>>(this->grid, storage, this->compact.data());
        this->compact.copyToHost(host.data());
        writeSlab(file, name, this->grid, host.data());
    };
    for (int d = 0; d < this->grid.num_dims; ++d) write(velocity_names[d], solver.velocity[d].data());
    write("pressure", solver.pressure.data());
    H5Fclose(file);

    this->saves.emplace_back(step, time);
    if (mpiRank() == 0) this->writeXdmf();
    this->flushForces();
}

void Output::writeXdmf() const {
    const GridView& g = this->grid;
    const bool is_3d = g.num_dims == 3;
    std::ostringstream cell_dims, node_dims;
    for (int d = g.num_dims - 1; d >= 0; --d) {
        cell_dims << g.global_cells[d] << (d > 0 ? " " : "");
        node_dims << g.global_cells[d] + 1 << (d > 0 ? " " : "");
    }
    auto data_item = [&](const std::string& dimensions, const std::string& source) {
        return "<DataItem Dimensions=\"" + dimensions + "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">" + source + "</DataItem>";
    };
    auto attribute = [&](const std::string& name, const std::string& source) {
        return "      <Attribute Name=\"" + name + "\" AttributeType=\"Scalar\" Center=\"Cell\">" + data_item(cell_dims.str(), source) + "</Attribute>\n";
    };

    std::ostringstream xdmf;
    xdmf << "<?xml version=\"1.0\" ?>\n<Xdmf Version=\"3.0\">\n  <Domain>\n"
         << "    <Grid Name=\"fields\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
    const char* node_names[max_dims] = {"x_nodes", "y_nodes", "z_nodes"};
    const char* velocity_names[max_dims] = {"u", "v", "w"};
    for (const auto& save : this->saves) {
        const std::string fields = stepFilename(save.first);
        xdmf << "    <Grid Name=\"step_" << save.first << "\" GridType=\"Uniform\">\n"
             << "      <Time Value=\"" << save.second << "\"/>\n"
             << "      <Topology TopologyType=\"" << (is_3d ? "3DRectMesh" : "2DRectMesh") << "\" Dimensions=\"" << node_dims.str() << "\"/>\n"
             << "      <Geometry GeometryType=\"" << (is_3d ? "VXVYVZ" : "VXVY") << "\">\n";
        for (int d = 0; d < g.num_dims; ++d)
            xdmf << "        " << data_item(std::to_string(g.global_cells[d] + 1), std::string("grid.h5:/") + node_names[d]) << "\n";
        xdmf << "      </Geometry>\n";
        for (int d = 0; d < g.num_dims; ++d) xdmf << attribute(velocity_names[d], fields + ":/" + velocity_names[d]);
        xdmf << attribute("pressure", fields + ":/pressure") << attribute("ib_markers", "grid.h5:/ib_markers") << "    </Grid>\n";
    }
    xdmf << "    </Grid>\n  </Domain>\n</Xdmf>\n";

    // write then rename, so a crash mid-write never leaves a broken index
    const std::string path = this->output_dir + "/fields.xmf";
    std::ofstream(path + ".tmp") << xdmf.str();
    std::rename((path + ".tmp").c_str(), path.c_str());
}

void Output::recordForces(int step, real time, const std::vector<real>& forces) {
    if (this->force_file == nullptr) return;
    std::ostringstream line;
    line.precision(10);
    line << std::scientific << step << " " << time;
    for (real value : forces) line << " " << value;
    this->force_lines.push_back(line.str());
}

void Output::flushForces() {
    if (this->force_file == nullptr) return;
    for (const std::string& line : this->force_lines) std::fprintf(this->force_file, "%s\n", line.c_str());
    std::fflush(this->force_file);
    this->force_lines.clear();
}
