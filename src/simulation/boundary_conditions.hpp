// domain boundaries as affine halo rules, halo = coefficient*mirrored interior + value, as in the prototype
#ifndef CTM_BOUNDARY_CONDITIONS_HPP
#define CTM_BOUNDARY_CONDITIONS_HPP

#include "mpi_proxy.hpp"

// field kinds that carry their own boundary rules
enum FieldKind : int { field_velocity_x = 0, field_velocity_y = 1, field_velocity_z = 2, field_pressure = 3, num_field_kinds = 4 };

struct HaloRule {
    real coefficient[max_dims][2];  // [direction][begin, end], +1 zero gradient, -1 zero value at the face
    real value[max_dims][2];

    // same coefficients with zero values, used by the linear operators and the pressure correction
    HaloRule homogeneous() const {
        HaloRule rule = *this;
        for (int d = 0; d < max_dims; ++d) rule.value[d][0] = rule.value[d][1] = 0.0;
        return rule;
    }
};

HaloRule makeHaloRule(const Parameters& parameters, FieldKind kind);

// fill every halo of a field: physical rules on the undecomposed axes, the neighbor exchange, then the decomposed axis ends
void fillHalos(const GridView& grid, const HaloRule& rule, HaloExchange& exchange, real* field);

#endif
