// ghost cell immersed boundary method, following MFC's m_ibm; Tseng & Ferziger (2003), Mittal et al. (2008)
#ifndef CTM_IBM_HPP
#define CTM_IBM_HPP

#include "boundary_conditions.hpp"
#include "geometry.hpp"

class ImmersedBoundaries {
public:
    ImmersedBoundaries(const Parameters& parameters, const GridView& grid);

    bool isActive() const { return this->is_active; }

    // mirror the image point velocity into each ghost cell and zero the body interior, for a stationary no-slip wall
    void applyGhostVelocity(real* velocity) const;

    // overwrite the ghost rows of an operator result: y_ghost = diagonal*(x_ghost + ghost_sign*x_image)
    void applyGhostRows(const real* x_storage, real* y_owned, real shift, real scale, const HaloRule& rule, real ghost_sign) const;

    // body force per unit density on each patch, the stress divergence summed over solid cells as in MFC
    std::vector<real> computeForces(const real* pressure, real* const* velocity, real kinematic_viscosity) const;

    GridView grid;
    Geometry geometry;

private:
    bool is_active;
};

#endif
