#include "geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#include "boundary_conditions.hpp"
#include "mpi_proxy.hpp"

namespace {

constexpr real tie_tolerance_fraction = 1.0e-10;  // relative to the model size, distance ties and on-surface points
constexpr real node_tolerance_fraction = 1.0e-12;  // relative to the cell width, image point on a cell center

using Triangle = std::array<real, 9>;

// ---------------------------------------------------------------- host: STL reading and transforms

bool readStlBinary(const std::string& path, std::vector<Triangle>& triangles, std::string& failure) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) { failure = "cannot open"; return false; }
    const long file_size = static_cast<long>(input.tellg());
    input.seekg(80);
    uint32_t num_triangles = 0;
    input.read(reinterpret_cast<char*>(&num_triangles), 4);
    if (file_size != 84 + 50*static_cast<long>(num_triangles)) { failure = "size does not match a binary STL"; return false; }
    triangles.resize(num_triangles);
    for (uint32_t t = 0; t < num_triangles; ++t) {
        float record[12];
        uint16_t attribute;
        input.read(reinterpret_cast<char*>(record), sizeof(record));
        input.read(reinterpret_cast<char*>(&attribute), sizeof(attribute));
        for (int v = 0; v < 9; ++v) triangles[t][v] = record[3 + v];  // skip the stored normal
    }
    return true;
}

bool readStlAscii(const std::string& path, std::vector<Triangle>& triangles, std::string& failure) {
    std::ifstream input(path);
    std::string word;
    std::vector<real> vertices;
    while (input >> word) {
        if (word != "vertex") continue;
        real x, y, z;
        input >> x >> y >> z;
        vertices.insert(vertices.end(), {x, y, z});
    }
    if (vertices.empty() || vertices.size() % 9 != 0) { failure = "vertex count is not a positive multiple of 3"; return false; }
    triangles.resize(vertices.size()/9);
    std::memcpy(triangles.data(), vertices.data(), vertices.size()*sizeof(real));
    return true;
}

// strictest reader first, collect every failure for the error message
std::vector<Triangle> readStl(const std::string& path) {
    std::vector<Triangle> triangles;
    std::string binary_failure, ascii_failure;
    if (readStlBinary(path, triangles, binary_failure)) return triangles;
    if (readStlAscii(path, triangles, ascii_failure)) return triangles;
    CTM_REQUIRE(false, "could not read %s (binary: %s, ascii: %s)", path.c_str(), binary_failure.c_str(), ascii_failure.c_str());
    return triangles;
}

// scale and rotate about the bounding box center, then translate, as MFC does
void transformTriangles(std::vector<Triangle>& triangles, const ImmersedPatch& patch) {
    real lower[3] = {1e300, 1e300, 1e300}, upper[3] = {-1e300, -1e300, -1e300};
    for (const Triangle& triangle : triangles)
        for (int v = 0; v < 3; ++v)
            for (int d = 0; d < 3; ++d) {
                lower[d] = std::min(lower[d], triangle[3*v + d]);
                upper[d] = std::max(upper[d], triangle[3*v + d]);
            }
    const real cx = std::cos(patch.rotate[0]), sx = std::sin(patch.rotate[0]);
    const real cy = std::cos(patch.rotate[1]), sy = std::sin(patch.rotate[1]);
    const real cz = std::cos(patch.rotate[2]), sz = std::sin(patch.rotate[2]);
    const real rotation_x[3][3] = {{1, 0, 0}, {0, cx, -sx}, {0, sx, cx}};
    const real rotation_y[3][3] = {{cy, 0, sy}, {0, 1, 0}, {-sy, 0, cy}};
    const real rotation_z[3][3] = {{cz, -sz, 0}, {sz, cz, 0}, {0, 0, 1}};
    real rotation_xy[3][3] = {}, rotation[3][3] = {};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int l = 0; l < 3; ++l) rotation_xy[r][c] += rotation_x[r][l]*rotation_y[l][c];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int l = 0; l < 3; ++l) rotation[r][c] += rotation_xy[r][l]*rotation_z[l][c];

    for (Triangle& triangle : triangles)
        for (int v = 0; v < 3; ++v) {
            real scaled[3];
            for (int d = 0; d < 3; ++d) scaled[d] = (triangle[3*v + d] - 0.5*(lower[d] + upper[d]))*patch.scale[d];
            for (int d = 0; d < 3; ++d)
                triangle[3*v + d] = rotation[d][0]*scaled[0] + rotation[d][1]*scaled[1] + rotation[d][2]*scaled[2]
                                    + 0.5*(lower[d] + upper[d]) + patch.translate[d];
        }
}

// outline edges (x0, y0, x1, y1) of a flat triangulation, counterclockwise so the winding number is positive inside
std::vector<real> outlineEdges(const std::vector<Triangle>& triangles) {
    std::vector<std::array<real, 4>> edges;
    real extent = 0.0;
    for (const Triangle& triangle : triangles) {
        real x[3] = {triangle[0], triangle[3], triangle[6]}, y[3] = {triangle[1], triangle[4], triangle[7]};
        if ((x[1] - x[0])*(y[2] - y[0]) - (y[1] - y[0])*(x[2] - x[0]) < 0) {
            std::swap(x[1], x[2]);
            std::swap(y[1], y[2]);
        }
        for (int e = 0; e < 3; ++e) {
            edges.push_back({x[e], y[e], x[(e + 1) % 3], y[(e + 1) % 3]});
            extent = std::max({extent, std::abs(x[e]), std::abs(y[e])});
        }
    }

    // an edge used by exactly one triangle lies on the outline
    const real tolerance = 1.0e-9*std::max(extent, 1e-300);
    std::map<std::array<long, 4>, int> counts;
    auto key = [&](const std::array<real, 4>& edge) {
        std::array<long, 4> quantized;
        for (int c = 0; c < 4; ++c) quantized[c] = std::lround(edge[c]/tolerance);
        if (std::make_pair(quantized[0], quantized[1]) > std::make_pair(quantized[2], quantized[3]))
            quantized = {quantized[2], quantized[3], quantized[0], quantized[1]};
        return quantized;
    };
    for (const auto& edge : edges) ++counts[key(edge)];
    std::vector<real> outline;
    for (const auto& edge : edges)
        if (counts[key(edge)] == 1) outline.insert(outline.end(), edge.begin(), edge.end());
    return outline;
}

// closed surface triangles with outward orientation, flipped when the signed volume is negative
std::vector<real> orientedTriangles(std::vector<Triangle> triangles) {
    real signed_volume = 0.0;
    for (const Triangle& t : triangles)
        signed_volume += t[0]*(t[4]*t[8] - t[5]*t[7]) - t[1]*(t[3]*t[8] - t[5]*t[6]) + t[2]*(t[3]*t[7] - t[4]*t[6]);
    std::vector<real> flat;
    for (Triangle& t : triangles) {
        if (signed_volume < 0) for (int d = 0; d < 3; ++d) std::swap(t[3 + d], t[6 + d]);
        flat.insert(flat.end(), t.begin(), t.end());
    }
    return flat;
}

// ---------------------------------------------------------------- device: geometry queries

struct GeometryView {
    int num_patches;
    int num_dims;
    const real* elements;        // 2D edges of 4 reals, 3D triangles of 9 reals
    const int* element_offsets;  // [num_patches + 1] in elements
    const real* bounding_boxes;  // [patch][lower xyz, upper xyz]
    const real* thresholds;
    const real* tie_tolerances;
};

struct Vector3 {
    real x, y, z;
};
__device__ inline Vector3 operator-(Vector3 a, Vector3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ inline Vector3 operator+(Vector3 a, Vector3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ inline Vector3 operator*(real s, Vector3 a) { return {s*a.x, s*a.y, s*a.z}; }
__device__ inline real dot(Vector3 a, Vector3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline Vector3 cross(Vector3 a, Vector3 b) { return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
__device__ inline real norm(Vector3 a) { return sqrt(dot(a, a)); }

__device__ inline Vector3 elementVertex(const real* element, int vertex, int num_dims) {
    return num_dims == 2 ? Vector3{element[2*vertex], element[2*vertex + 1], 0.0}
                         : Vector3{element[3*vertex], element[3*vertex + 1], element[3*vertex + 2]};
}

// closest point on a triangle, Ericson (2005) section 5.1.5
__device__ Vector3 closestPointOnTriangle(Vector3 point, Vector3 a, Vector3 b, Vector3 c) {
    const Vector3 ab = b - a, ac = c - a, ap = point - a;
    const real d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return a;
    const Vector3 bp = point - b;
    const real d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return b;
    const real vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + (d1/(d1 - d3))*ab;
    const Vector3 cp = point - c;
    const real d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;
    const real vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + (d2/(d2 - d6))*ac;
    const real va = d3*d6 - d5*d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return b + ((d4 - d3)/((d4 - d3) + (d5 - d6)))*(c - b);
    const real inverse = 1.0/(va + vb + vc);
    return a + (vb*inverse)*ab + (vc*inverse)*ac;
}

__device__ Vector3 closestPointOnElement(const GeometryView& geometry, int element, Vector3 point) {
    if (geometry.num_dims == 2) {
        const real* edge = geometry.elements + 4*element;
        const Vector3 start = elementVertex(edge, 0, 2), end = elementVertex(edge, 1, 2);
        const Vector3 along = end - start;
        const real fraction = fmin(fmax(dot(point - start, along)/fmax(dot(along, along), 1e-300), 0.0), 1.0);
        return start + fraction*along;
    }
    const real* triangle = geometry.elements + 9*element;
    return closestPointOnTriangle(point, elementVertex(triangle, 0, 3), elementVertex(triangle, 1, 3), elementVertex(triangle, 2, 3));
}

__device__ Vector3 outwardElementNormal(const GeometryView& geometry, int element) {
    if (geometry.num_dims == 2) {
        const real* edge = geometry.elements + 4*element;
        const Vector3 normal = {edge[3] - edge[1], -(edge[2] - edge[0]), 0.0};
        return (1.0/norm(normal))*normal;
    }
    const real* triangle = geometry.elements + 9*element;
    const Vector3 a = elementVertex(triangle, 0, 3);
    const Vector3 normal = cross(elementVertex(triangle, 1, 3) - a, elementVertex(triangle, 2, 3) - a);
    return (1.0/norm(normal))*normal;
}

// generalized winding number, Jacobson et al. (2013): edge angles in 2D, solid angles of Van Oosterom & Strackee (1983) in 3D
__device__ real windingNumber(const GeometryView& geometry, int patch, Vector3 point) {
    real winding = 0.0;
    for (int element = geometry.element_offsets[patch]; element < geometry.element_offsets[patch + 1]; ++element) {
        if (geometry.num_dims == 2) {
            const real* edge = geometry.elements + 4*element;
            const Vector3 to_start = elementVertex(edge, 0, 2) - point, to_end = elementVertex(edge, 1, 2) - point;
            winding += atan2(to_start.x*to_end.y - to_start.y*to_end.x, to_start.x*to_end.x + to_start.y*to_end.y);
        } else {
            const real* triangle = geometry.elements + 9*element;
            const Vector3 a = elementVertex(triangle, 0, 3) - point, b = elementVertex(triangle, 1, 3) - point, c = elementVertex(triangle, 2, 3) - point;
            const real length_a = norm(a), length_b = norm(b), length_c = norm(c);
            if (length_a*length_b*length_c < 1e-300) continue;
            winding += atan2(dot(a, cross(b, c)), length_a*length_b*length_c + dot(a, b)*length_c + dot(b, c)*length_a + dot(c, a)*length_b);
        }
    }
    return fabs(winding/(2.0*pi));
}

__device__ inline Vector3 cellCenter(const GridView& grid, int i, int j, int k) {
    return {grid.center(0, i), grid.center(1, j), numDims(grid) == 3 ? grid.center(2, k) : 0.0};
}

__global__ void markersKernel(GridView grid, GeometryView geometry, int* markers) {
    const long cell = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (cell >= grid.numStorage()) return;
    const int i = static_cast<int>(cell % grid.storage[0]) - grid.halo[0];
    const int j = static_cast<int>((cell/grid.storage[0]) % grid.storage[1]) - grid.halo[1];
    const int k = static_cast<int>(cell/(static_cast<long>(grid.storage[0])*grid.storage[1])) - grid.halo[2];
    const Vector3 center = cellCenter(grid, i, j, k);
    int marker = 0;
    for (int patch = 0; patch < geometry.num_patches; ++patch) {
        const real* box = geometry.bounding_boxes + 6*patch;
        const bool in_box = center.x >= box[0] && center.x <= box[3] && center.y >= box[1] && center.y <= box[4] &&
                            (geometry.num_dims == 2 || (center.z >= box[2] && center.z <= box[5]));
        if (in_box && windingNumber(geometry, patch, center) > geometry.thresholds[patch]) marker = patch + 1;
    }
    markers[cell] = marker;
}

// fluid cells on any domain face can reach the outer flow
__global__ void seedReachableKernel(GridView grid, const int* markers, real* reachable) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int index[max_dims];
    grid.ownedCell(owned, index[0], index[1], index[2]);
    const long cell = grid.storageIndex(index[0], index[1], index[2]);
    bool on_face = false;
    for (int d = 0; d < numDims(grid); ++d) {
        const int global_index = index[d] + grid.offset[d];
        on_face |= global_index == 0 || global_index == grid.global_cells[d] - 1;
    }
    reachable[cell] = markers[cell] == 0 && on_face ? 1.0 : 0.0;
}

// spread reachability to face neighboring fluid cells, monotonic so the in place update is safe
__global__ void growReachableKernel(GridView grid, const int* markers, real* reachable, int* changed) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int index[max_dims];
    grid.ownedCell(owned, index[0], index[1], index[2]);
    const long cell = grid.storageIndex(index[0], index[1], index[2]);
    if (markers[cell] != 0 || reachable[cell] > 0.0) return;
    for (int d = 0; d < numDims(grid); ++d) {
        for (int side = -1; side <= 1; side += 2) {
            const int neighbor_global = index[d] + side + grid.offset[d];
            if (!grid.periodic[d] && (neighbor_global < 0 || neighbor_global >= grid.global_cells[d])) continue;
            if (reachable[cell + side*grid.stride(d)] > 0.0) {
                reachable[cell] = 1.0;
                *changed = 1;
                return;
            }
        }
    }
}

// fluid the outer flow cannot reach becomes part of a neighboring body
__global__ void fillTrappedKernel(GridView grid, int* markers, const real* reachable, int* num_filled) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int index[max_dims];
    grid.ownedCell(owned, index[0], index[1], index[2]);
    const long cell = grid.storageIndex(index[0], index[1], index[2]);
    if (markers[cell] != 0 || reachable[cell] > 0.0) return;
    int patch = 1;
    for (int d = 0; d < numDims(grid); ++d)
        for (int side = -1; side <= 1; side += 2) patch = markers[cell + side*grid.stride(d)] != 0 ? markers[cell + side*grid.stride(d)] : patch;
    markers[cell] = patch;
    atomicAdd(num_filled, 1);
}

__global__ void markersToRealKernel(long count, const int* markers, real* values) {
    const long cell = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (cell < count) values[cell] = markers[cell];
}

__global__ void realToMarkersKernel(long count, const real* values, int* markers) {
    const long cell = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (cell < count) markers[cell] = static_cast<int>(lround(values[cell]));
}

// a solid cell with fluid within gp_layers cells in each direction is a ghost point
__global__ void cellTypesKernel(GridView grid, const int* markers, int gp_layers, signed char* cell_types, int* ghost_counter) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    if (markers[grid.storageIndex(i, j, k)] == 0) {
        cell_types[owned] = cell_fluid;
        return;
    }
    const int reach_z = numDims(grid) == 3 ? gp_layers : 0;
    bool has_fluid = false;
    for (int dk = -reach_z; dk <= reach_z; ++dk)
        for (int dj = -gp_layers; dj <= gp_layers; ++dj)
            for (int di = -gp_layers; di <= gp_layers; ++di)
                has_fluid |= markers[grid.storageIndex(i + di, j + dj, k + dk)] == 0;
    cell_types[owned] = has_fluid ? cell_ghost : cell_interior;
    if (has_fluid) atomicAdd(ghost_counter, 1);
}

__global__ void listGhostPointsKernel(GridView grid, const int* markers, const signed char* cell_types, int* slot_counter,
                                      long* storage_cells, long* owned_cells, int* patch_ids) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned() || cell_types[owned] != cell_ghost) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    const int slot = atomicAdd(slot_counter, 1);  // order does not matter, ghost points are independent
    storage_cells[slot] = grid.storageIndex(i, j, k);
    owned_cells[slot] = owned;
    patch_ids[slot] = markers[storage_cells[slot]];
}

// levelset, outward normal averaged over tied elements, mirrored image point, and its interpolation weights
__global__ void imagePointsKernel(GridView grid, GeometryView geometry, const int* markers, int gp_layers, int num_ghost_points,
                                  const long* owned_cells, const int* patch_ids, long* stencil_cells, real* weights, int* bilinear_counter,
                                  int* error_flag, int* fallback_counter) {
    const int ghost = blockIdx.x*blockDim.x + threadIdx.x;
    if (ghost >= num_ghost_points) return;
    int i, j, k;
    grid.ownedCell(owned_cells[ghost], i, j, k);
    const int num_dims = numDims(grid);
    const int patch = patch_ids[ghost] - 1;
    const Vector3 center = cellCenter(grid, i, j, k);
    const real tolerance = geometry.tie_tolerances[patch];

    // nearest distance, then the mean direction to every element within the tie tolerance
    real nearest = 1e300;
    for (int element = geometry.element_offsets[patch]; element < geometry.element_offsets[patch + 1]; ++element)
        nearest = fmin(nearest, norm(closestPointOnElement(geometry, element, center) - center));
    Vector3 direction = {0.0, 0.0, 0.0}, first_direction = {0.0, 0.0, 0.0};
    for (int element = geometry.element_offsets[patch]; element < geometry.element_offsets[patch + 1]; ++element) {
        const Vector3 offset = closestPointOnElement(geometry, element, center) - center;
        const real distance = norm(offset);
        if (distance > nearest + tolerance) continue;
        const Vector3 element_direction = distance <= tolerance ? outwardElementNormal(geometry, element) : (1.0/distance)*offset;
        if (norm(first_direction) == 0.0) first_direction = element_direction;
        direction = direction + element_direction;
    }
    // opposite faces of a feature thinner than two cells cancel, keep one of them instead
    if (norm(direction) < 0.5) direction = first_direction;
    const Vector3 normal = (1.0/norm(direction))*direction;
    const real image[max_dims] = {center.x + 2.0*nearest*normal.x, center.y + 2.0*nearest*normal.y, center.z + 2.0*nearest*normal.z};

    // lower corner of the bracketing cell centers and the fractional position inside it
    int lower[max_dims] = {0, 0, 0};
    real fraction[max_dims] = {0.0, 0.0, 0.0};
    for (int d = 0; d < num_dims; ++d) {
        const real position = (image[d] - grid.origin[d])/grid.spacing[d] - 0.5;
        const int lower_global = isfinite(position) ? static_cast<int>(floor(position)) : -1;
        if (lower_global < 0 || lower_global + 1 >= grid.global_cells[d] ||
            lower_global - grid.offset[d] < -grid.halo[d] || lower_global + 1 - grid.offset[d] >= grid.cells[d] + grid.halo[d]) {
            atomicExch(error_flag, 1);
            return;
        }
        lower[d] = lower_global - grid.offset[d];
        fraction[d] = fmin(fmax(position - lower_global, 0.0), 1.0);
    }
    lower[2] = num_dims == 3 ? lower[2] : k;

    // trilinear on an all fluid stencil, Mittal et al. (2008), else inverse distance squared on the fluid cells, Chaudhuri et al. (2011)
    const int num_corners = num_dims == 3 ? 8 : 4;
    long corner_cells[max_stencil];
    bool is_fluid[max_stencil];
    real trilinear[max_stencil], inverse_distance[max_stencil];
    bool all_fluid = true, on_node = false;
    const real node_tolerance = node_tolerance_fraction*fmin(grid.spacing[0], grid.spacing[1]);
    for (int corner = 0; corner < num_corners; ++corner) {
        const int offset[max_dims] = {corner & 1, (corner >> 1) & 1, (corner >> 2) & 1};
        corner_cells[corner] = grid.storageIndex(lower[0] + offset[0], lower[1] + offset[1], lower[2] + offset[2]);
        is_fluid[corner] = markers[corner_cells[corner]] == 0;
        all_fluid &= is_fluid[corner];
        trilinear[corner] = 1.0;
        real distance_squared = 0.0;
        for (int d = 0; d < num_dims; ++d) {
            trilinear[corner] *= offset[d] ? fraction[d] : 1.0 - fraction[d];
            const real separation = grid.center(d, lower[d] + offset[d]) - image[d];
            distance_squared += separation*separation;
        }
        const real distance = sqrt(distance_squared);
        inverse_distance[corner] = distance <= node_tolerance ? -1.0 : 1.0/(distance*distance);  // -1 marks an exact node
        on_node |= distance <= node_tolerance && is_fluid[corner];
    }
    real weight_sum = 0.0;
    for (int corner = 0; corner < num_corners; ++corner) {
        real weight = all_fluid ? trilinear[corner] : on_node ? (inverse_distance[corner] < 0 ? 1.0 : 0.0) : inverse_distance[corner];
        weight = is_fluid[corner] ? weight : 0.0;
        weights[ghost*max_stencil + corner] = weight;
        weight_sum += weight;
    }
    // a stencil buried in the body falls back on the ghost point's own fluid neighbors nearest the image point, as MFC does
    if (weight_sum <= 0.0) {
        atomicAdd(fallback_counter, 1);
        real nearest_distances[max_stencil];
        int num_found = 0;
        const int reach_z = num_dims == 3 ? gp_layers : 0;
        for (int dk = -reach_z; dk <= reach_z; ++dk)
            for (int dj = -gp_layers; dj <= gp_layers; ++dj)
                for (int di = -gp_layers; di <= gp_layers; ++di) {
                    const long cell = grid.storageIndex(i + di, j + dj, k + dk);
                    if (markers[cell] != 0) continue;
                    const real separation[max_dims] = {grid.center(0, i + di) - image[0], grid.center(1, j + dj) - image[1],
                                                       num_dims == 3 ? grid.center(2, k + dk) - image[2] : 0.0};
                    const real distance = fmax(sqrt(separation[0]*separation[0] + separation[1]*separation[1] + separation[2]*separation[2]),
                                               node_tolerance);
                    int slot;
                    if (num_found < max_stencil) {
                        slot = num_found++;
                    } else if (distance < nearest_distances[max_stencil - 1]) {
                        slot = max_stencil - 1;  // replace the farthest kept neighbor
                    } else {
                        continue;
                    }
                    while (slot > 0 && nearest_distances[slot - 1] > distance) {  // keep the list sorted by distance
                        nearest_distances[slot] = nearest_distances[slot - 1];
                        corner_cells[slot] = corner_cells[slot - 1];
                        --slot;
                    }
                    nearest_distances[slot] = distance;
                    corner_cells[slot] = cell;
                }
        for (int corner = 0; corner < max_stencil; ++corner) {
            const bool is_used = corner < num_found;
            stencil_cells[ghost*max_stencil + corner] = is_used ? corner_cells[corner] : corner_cells[0];
            weights[ghost*max_stencil + corner] = is_used ? 1.0/(nearest_distances[corner]*nearest_distances[corner]) : 0.0;
            weight_sum += weights[ghost*max_stencil + corner];
        }
        if (num_found == 0) atomicExch(error_flag, 2);
        for (int corner = 0; corner < max_stencil; ++corner) weights[ghost*max_stencil + corner] /= fmax(weight_sum, 1e-300);
        return;
    }
    for (int corner = 0; corner < max_stencil; ++corner) {
        stencil_cells[ghost*max_stencil + corner] = corner < num_corners ? corner_cells[corner] : corner_cells[0];
        weights[ghost*max_stencil + corner] = corner < num_corners ? weights[ghost*max_stencil + corner]/weight_sum : 0.0;
        if (!isfinite(weights[ghost*max_stencil + corner])) atomicExch(error_flag, 3);
    }
    if (all_fluid) atomicAdd(bilinear_counter, 1);
}

}  // namespace

Geometry::Geometry(const Parameters& parameters, const GridView& grid) {
    this->markers.resize(grid.numStorage());
    this->cell_types.resize(grid.numOwned());
    if (!parameters.ib) return;
    this->num_patches = static_cast<int>(parameters.immersed_patches.size());

    // read and place every model on the host
    std::vector<real> elements, bounding_boxes, thresholds, tie_tolerances;
    std::vector<int> element_offsets = {0};
    const int values_per_element = parameters.num_dims == 2 ? 4 : 9;
    for (const ImmersedPatch& patch : parameters.immersed_patches) {
        std::vector<Triangle> triangles = readStl(patch.stl_path);
        transformTriangles(triangles, patch);
        const std::vector<real> model = parameters.num_dims == 2 ? outlineEdges(triangles) : orientedTriangles(triangles);
        CTM_REQUIRE(!model.empty(), "%s has no outline or surface", patch.stl_path.c_str());
        real lower[3] = {1e300, 1e300, 1e300}, upper[3] = {-1e300, -1e300, -1e300};
        const int coordinates = parameters.num_dims;
        for (size_t v = 0; v < model.size(); v += coordinates)
            for (int d = 0; d < coordinates; ++d) {
                lower[d] = std::min(lower[d], model[v + d]);
                upper[d] = std::max(upper[d], model[v + d]);
            }
        if (coordinates == 2) lower[2] = upper[2] = 0.0;
        elements.insert(elements.end(), model.begin(), model.end());
        element_offsets.push_back(static_cast<int>(elements.size()/values_per_element));
        bounding_boxes.insert(bounding_boxes.end(), {lower[0], lower[1], lower[2], upper[0], upper[1], upper[2]});
        thresholds.push_back(patch.threshold);
        tie_tolerances.push_back(tie_tolerance_fraction*std::max({upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]}));
        if (mpiRank() == 0)
            std::printf("INFO :: %s: %d %s\n", patch.stl_path.c_str(), element_offsets.back() - element_offsets[element_offsets.size() - 2],
                        parameters.num_dims == 2 ? "outline edges" : "triangles");
    }
    DeviceArray<real> device_elements, device_boxes, device_thresholds, device_tolerances;
    DeviceArray<int> device_offsets;
    device_elements.copyFromHost(elements.data(), elements.size());
    device_boxes.copyFromHost(bounding_boxes.data(), bounding_boxes.size());
    device_thresholds.copyFromHost(thresholds.data(), thresholds.size());
    device_tolerances.copyFromHost(tie_tolerances.data(), tie_tolerances.size());
    device_offsets.copyFromHost(element_offsets.data(), element_offsets.size());
    const GeometryView view = {this->num_patches, parameters.num_dims, device_elements.data(), device_offsets.data(),
                               device_boxes.data(), device_thresholds.data(), device_tolerances.data()};

    // solid markers on every stored cell, so neighbors in the halo are known without an exchange
    markersKernel<<<numBlocks(grid.numStorage()), block_size>>>(grid, view, this->markers.data());
    CUDA_CHECK(cudaGetLastError());
    this->fillTrappedFluid(grid);

    // classify owned cells and list the ghost points
    DeviceArray<int> counters(5);  // ghost count, slot, bilinear count, error flag, neighbor fallback count
    cellTypesKernel<<<numBlocks(grid.numOwned()), block_size>>>(grid, this->markers.data(), parameters.gp_layers, this->cell_types.data(),
                                                                 counters.data());
    int host_counters[5];
    counters.copyToHost(host_counters);
    GhostPoints& ghosts = this->ghost_points;
    ghosts.count = host_counters[0];
    ghosts.storage_cells.resize(ghosts.count);
    ghosts.owned_cells.resize(ghosts.count);
    ghosts.patch_ids.resize(ghosts.count);
    ghosts.stencil_cells.resize(static_cast<long>(ghosts.count)*max_stencil);
    ghosts.weights.resize(static_cast<long>(ghosts.count)*max_stencil);
    listGhostPointsKernel<<<numBlocks(grid.numOwned()), block_size>>>(grid, this->markers.data(), this->cell_types.data(), counters.data() + 1,
                                                                       ghosts.storage_cells.data(), ghosts.owned_cells.data(), ghosts.patch_ids.data());
    if (ghosts.count > 0)
        imagePointsKernel<<<numBlocks(ghosts.count), block_size>>>(grid, view, this->markers.data(), parameters.gp_layers, ghosts.count,
                                                                    ghosts.owned_cells.data(), ghosts.patch_ids.data(), ghosts.stencil_cells.data(),
                                                                    ghosts.weights.data(), counters.data() + 2, counters.data() + 3,
                                                                    counters.data() + 4);
    CUDA_CHECK(cudaGetLastError());
    counters.copyToHost(host_counters);
    CTM_REQUIRE(host_counters[3] != 1, "an image point stencil leaves this rank or the domain, refine the grid or move the body");
    CTM_REQUIRE(host_counters[3] != 2, "a ghost point has no fluid neighbor within gp_layers");
    CTM_REQUIRE(host_counters[3] != 3, "an image point interpolation weight is not finite");

    real totals[3] = {static_cast<real>(ghosts.count), static_cast<real>(host_counters[2]), static_cast<real>(host_counters[4])};
    allreduceSum(totals, 3);
    this->global_ghost_points = static_cast<long>(totals[0]);
    this->global_bilinear = static_cast<long>(totals[1]);
    this->global_fallback = static_cast<long>(totals[2]);
}

// fluid pockets sealed by the bodies have no pressure level and make the Poisson problem singular, so they become solid
void Geometry::fillTrappedFluid(const GridView& grid) {
    HaloExchange exchange(grid);
    HaloRule copy_rule;
    for (int d = 0; d < max_dims; ++d)
        for (int side = 0; side < 2; ++side) copy_rule.coefficient[d][side] = 1.0, copy_rule.value[d][side] = 0.0;
    DeviceArray<real> reachable(grid.numStorage());
    DeviceArray<int> flags(2);  // changed, number filled
    seedReachableKernel<<<numBlocks(grid.numOwned()), block_size>>>(grid, this->markers.data(), reachable.data());

    constexpr int sweeps_per_exchange = 8;
    for (int changed = 1; changed != 0;) {
        flags.fillZero();
        for (int sweep = 0; sweep < sweeps_per_exchange; ++sweep)
            growReachableKernel<<<numBlocks(grid.numOwned()), block_size>>>(grid, this->markers.data(), reachable.data(), flags.data());
        fillHalos(grid, copy_rule, exchange, reachable.data());
        int host_flags[2];
        flags.copyToHost(host_flags);
        changed = static_cast<int>(allreduceMax(static_cast<real>(host_flags[0])));
    }

    flags.fillZero();
    fillTrappedKernel<<<numBlocks(grid.numOwned()), block_size>>>(grid, this->markers.data(), reachable.data(), flags.data() + 1);
    int host_flags[2];
    flags.copyToHost(host_flags);
    this->global_trapped = static_cast<long>(allreduceSum(static_cast<real>(host_flags[1])));
    if (this->global_trapped == 0) return;

    // neighbor ranks see the filled cells in their halos
    markersToRealKernel<<<numBlocks(grid.numStorage()), block_size>>>(grid.numStorage(), this->markers.data(), reachable.data());
    exchange.exchange(reachable.data());
    realToMarkersKernel<<<numBlocks(grid.numStorage()), block_size>>>(grid.numStorage(), reachable.data(), this->markers.data());
    CUDA_CHECK(cudaGetLastError());
}
