#pragma once
#include <Eigen/Core>
#include <memory>
#include <span>

namespace xmvb::integrals::nao {
// R_l(r) = r^l f(r). Supply reduced f on a strictly increasing grid starting
// at zero and ending at the compact-support radius. Units: bohr, Hartree.
// f is regular with f'(0)=0; f(rc)=f'(rc)=0. No implicit normalization.
// Real harmonics follow ABACUS's Condon-Shortley convention (pz,-px,-py).
class Radial {
  public:
    Radial(int l, std::span<const double> grid, std::span<const double> reduced_values);
    ~Radial();
    Radial(const Radial&) = delete;
    Radial& operator=(const Radial&) = delete;
    int l() const;
    double cutoff() const;
    double value(const Eigen::Vector3d& displacement, int m) const;
    double kinetic(const Eigen::Vector3d& displacement, int m) const;
    double potential(const Eigen::Vector3d& displacement, int m) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace xmvb::integrals::nao
