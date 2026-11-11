#pragma once

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <vector>

#include "vbscf/optimization/coupled/affine.hpp"

namespace xmvb::vb {

/** A Galerkin restriction of the accepted-point affine CI tangent.
 *
 * Only Z, C Z, M Z and B^T Z are retained. Davidson consumes their small
 * projections without applying a CI operator to each orbital direction.
 */
class CiahCiResponseSpace {
public:
    CiahCiResponseSpace(const StructureTangentOperator& structure,
                        const AffineCoupling& coupling,
                        Eigen::Index orbital_size,
                        Eigen::Index capacity)
        : structure_(structure), coupling_(coupling),
          zero_orbital_(Eigen::VectorXd::Zero(orbital_size)),
          h_(Eigen::MatrixXd::Zero(capacity, capacity)), m_(h_) {}

    Eigen::Index rank() const { return static_cast<Eigen::Index>(columns_.size()); }

    Eigen::VectorXd forcing(const Eigen::VectorXd& p) const {
        return coupling_.fixed_overlap_forcing(p);
    }

    std::pair<Eigen::VectorXd, Eigen::VectorXd> orbital_images(const Eigen::VectorXd& p) const {
        return coupling_.fixed_overlap_orbital_images(p, false);
    }

    Eigen::VectorXd lift(const Eigen::VectorXd& y) const {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(structure_.tangent_size());
        for (Eigen::Index i = 0; i < rank(); ++i)
            q.noalias() += y[i] * columns_[i].z;
        return q;
    }

    Eigen::VectorXd project(const Eigen::VectorXd& g) const {
        Eigen::VectorXd y = Eigen::VectorXd::Zero(h_.rows());
        for (Eigen::Index i = 0; i < rank(); ++i)
            y[i] = columns_[i].z.dot(g);
        return y;
    }

    void add_hessian_image(const Eigen::VectorXd& p, const Eigen::VectorXd& y,
                           Eigen::Ref<Eigen::VectorXd> hp,
                           Eigen::Ref<Eigen::VectorXd> hy) const {
        hy.setZero();
        hy.head(rank()).noalias() = h_.topLeftCorner(rank(), rank()) * y.head(rank());
        for (Eigen::Index i = 0; i < rank(); ++i) {
            hp.noalias() += y[i] * columns_[i].adjoint;
            hy[i] += columns_[i].adjoint.dot(p);
        }
    }

    double hessian_coordinate(Eigen::Index j,
                              const Eigen::Ref<const Eigen::VectorXd>& p,
                              const Eigen::Ref<const Eigen::VectorXd>& y) const {
        return columns_[j].adjoint.dot(p) +
               h_.row(j).head(rank()).dot(y.head(rank()));
    }

    bool bounded_by(const Eigen::Ref<const Eigen::VectorXd>& y,
                    double scaled_limit, double relative_error) const {
        const auto active = y.head(rank());
        if (active.isZero(0.0))
            return true;
        const auto gram = m_.topLeftCorner(rank(), rank());
        const double squared = active.dot(gram * active);
        const auto magnitude = active.cwiseAbs();
        const double scale = magnitude.dot(gram.cwiseAbs() * magnitude);
        // The coefficient infinity norm is bounded by the physical 2-norm.
        // Near the comparison boundary, defer to the direct coefficient maximum.
        return squared > 0.0 && squared + relative_error * scale < scaled_limit * scaled_limit;
    }

    Eigen::VectorXd metric(const Eigen::VectorXd& y) const {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(m_.rows());
        result.head(rank()).noalias() = m_.topLeftCorner(rank(), rank()) * y.head(rank());
        return result;
    }

    Eigen::VectorXd omitted_residual(Eigen::VectorXd rhs,
                                     const Eigen::VectorXd& y,
                                     double omega) const {
        add_residual_image(rhs, y, omega);
        for (const auto& column : columns_)
            rhs.noalias() -= column.z.dot(rhs) * column.z;
        return rhs;
    }

    bool add(const Eigen::VectorXd& residual, double omega, double lindep) {
        Eigen::VectorXd z =
            structure_.apply_inverse_shifted_preconditioner(residual, std::max(0.0, -omega));
        const double initial_norm = z.stableNorm();
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& column : columns_)
                z.noalias() -= column.z.dot(z) * column.z;
        const double norm = z.stableNorm();
        if (!(norm > lindep * initial_norm))
            return false;
        z /= norm;
        auto image = coupling_.apply(zero_orbital_, z);
        const Eigen::Index j = rank();
        for (Eigen::Index i = 0; i < j; ++i) {
            h_(i, j) = h_(j, i) = columns_[i].z.dot(image.ci.hessian_coordinates);
            m_(i, j) = m_(j, i) = columns_[i].z.dot(image.ci.metric_coordinates);
        }
        h_(j, j) = z.dot(image.ci.hessian_coordinates);
        m_(j, j) = z.dot(image.ci.metric_coordinates);
        columns_.push_back({std::move(z),
                            std::move(image.ci.hessian_coordinates),
                            std::move(image.ci.metric_coordinates),
                            std::move(image.adjoint)});
        return true;
    }

    void discard_last_column() { columns_.pop_back(); }

    /** Reuse a single forward Bp while completing the shifted CI residual.
     * Indefinite projected CI curvature stays in the joint AH solve.
     */
    bool residual(const Eigen::VectorXd& rhs, double omega, Eigen::VectorXd* r,
                  Eigen::VectorXd* orbital_residual = nullptr) const {
        if (r)
            *r = rhs;
        if (rank() == 0)
            return true;
        const Eigen::MatrixXd shifted =
            h_.topLeftCorner(rank(), rank()) - omega * m_.topLeftCorner(rank(), rank());
        const Eigen::LLT<Eigen::MatrixXd> factor(shifted);
        if (factor.info() != Eigen::Success)
            return false;
        const Eigen::VectorXd y = factor.solve(-project(rhs).head(rank()));
        if (r)
            add_residual_image(*r, y, omega);
        if (orbital_residual)
            for (Eigen::Index i = 0; i < rank(); ++i)
                orbital_residual->noalias() += y[i] * columns_[i].adjoint;
        return true;
    }

private:
    void add_residual_image(Eigen::VectorXd& residual,
                            const Eigen::VectorXd& y,
                            double omega) const {
        for (Eigen::Index i = 0; i < rank(); ++i) {
            residual.noalias() += y[i] * columns_[i].h;
            if (omega != 0.0)
                residual.noalias() -= (omega * y[i]) * columns_[i].m;
        }
    }

    struct Column {
        Eigen::VectorXd z, h, m, adjoint;
    };
    const StructureTangentOperator& structure_;
    const AffineCoupling& coupling_;
    Eigen::VectorXd zero_orbital_;
    Eigen::MatrixXd h_, m_;
    std::vector<Column> columns_;
};

} // namespace xmvb::vb
