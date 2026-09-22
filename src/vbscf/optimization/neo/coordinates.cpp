#include "vbscf/optimization/neo/coordinates.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/orbitals/charts/chart.hpp"

namespace xmvb::vb {

CoupledNeoCoordinates::CoupledNeoCoordinates(
    const OrbitalChart& orbital_chart,
    const NonredundantRetractionMetric& orbital_metric,
    const StructureTangentOperator& structure)
    : orbital_metric_(&orbital_metric),
      structure_(&structure),
      orbital_size_(orbital_chart.reduced_size()),
      structure_size_(structure.tangent_size()) {
  if (orbital_size_ < 0 || structure_size_ < 0) {
    throw std::invalid_argument("invalid coupled NEO tangent dimensions");
  }
}

Eigen::VectorXd CoupledNeoCoordinates::flatten(
    const CoupledDirection& direction) const {
  if (direction.orbital.size() != orbital_size_ ||
      !direction.orbital.allFinite()) {
    throw std::invalid_argument(
        "coupled NEO orbital direction has incompatible dimensions or values");
  }

  Eigen::VectorXd coordinates(size());
  coordinates.head(orbital_size_) = direction.orbital;
  coordinates.tail(structure_size_) =
      structure_->coordinates(direction.structure);
  return coordinates;
}

CoupledDirection CoupledNeoCoordinates::unflatten(
    const Eigen::VectorXd& coordinates) const {
  if (coordinates.size() != size() || !coordinates.allFinite()) {
    throw std::invalid_argument(
        "coupled NEO coordinates have incompatible dimensions or values");
  }
  CoupledDirection direction;
  direction.orbital = coordinates.head(orbital_size_);
  direction.structure = structure_->expand(coordinates.tail(structure_size_));
  return direction;
}

Eigen::VectorXd CoupledNeoCoordinates::apply_metric(
    const Eigen::VectorXd& coordinates) const {
  if (coordinates.size() != size() || !coordinates.allFinite()) {
    throw std::invalid_argument(
        "coupled NEO coordinates have incompatible dimensions or values");
  }
  Eigen::VectorXd image(size());
  image.head(orbital_size_) =
      orbital_metric_->apply(coordinates.head(orbital_size_));
  image.tail(structure_size_) =
      structure_->apply_metric_coordinates(coordinates.tail(structure_size_));
  return image;
}

double CoupledNeoCoordinates::squared_norm(
    const Eigen::VectorXd& coordinates) const {
  const Eigen::VectorXd metric_coordinates = apply_metric(coordinates);
  const double norm = coordinates.dot(metric_coordinates);
  const double tolerance = 1024.0 * std::numeric_limits<double>::epsilon() *
      std::max<Eigen::Index>(1, size()) *
      std::max(1.0, coordinates.squaredNorm());
  if (!std::isfinite(norm) || norm < -tolerance) {
    throw std::runtime_error("coupled NEO metric is not positive definite");
  }
  return std::max(0.0, norm);
}

}  // namespace xmvb::vb
