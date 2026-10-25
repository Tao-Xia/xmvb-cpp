#include <iostream>
#include <memory>
#include <stdexcept>
#include <Eigen/SparseCore>
#include "spinor_reference.hpp"
#include "vbscf/orbitals/spinor/frame.hpp"
#include "vbscf/structures/spinor/action.hpp"

namespace {
using namespace xmvb::vb::spinor;
using Complex = std::complex<double>;

void close(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols() ||
        (a - b).norm() > 2e-11 * std::max(1.0, b.norm())) {
        throw std::runtime_error("spinor projection reference mismatch");
    }
}

template <class Function> void rejects(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::length_error&) {
        return;
    }
    throw std::runtime_error("invalid spinor input accepted");
}

void test_projection(int n, int electrons) {
    auto strings = std::make_shared<Strings>(n, electrons, n / 2, 1000);
    const int d = strings->dimension();
    Eigen::MatrixXcd r = Eigen::MatrixXcd::Zero(n, n);
    for (int q = 0; q < n; ++q) {
        r(q, q) = 1.0 + 0.13 * q;
        for (int p = 0; p < q; ++p)
            r(p, q) = Complex(0.07 * (p + 1), -0.09 * (q + 1));
    }
    Transform transform(strings, r);
    const Eigen::MatrixXcd t = spinor_reference::wedge_matrix(r, strings->masks());
    Eigen::MatrixXcd x = spinor_reference::deterministic_complex(d, 3);
    Eigen::MatrixXcd y = x;
    transform.apply(y);
    close(y, t * x);
    y = x;
    transform.apply_adjoint(y);
    close(y, t.adjoint() * x);

    const int selected = std::min(3, d);
    Eigen::MatrixXcd b = Eigen::MatrixXcd::Zero(d, selected);
    for (int j = 0; j < selected; ++j)
        b(j, j) = 1.0;
    if (d > selected)
        b(d - 1, 0) = Complex(0.3, 0.2);
    Eigen::SparseMatrix<Complex> mapping = b.sparseView();
    Basis basis(strings, mapping);
    Eigen::MatrixXcd raw = spinor_reference::deterministic_complex(n, n);
    Eigen::MatrixXcd h = (raw + raw.adjoint()).eval();
    Eigen::MatrixXcd l = (0.1 * h).eval();
    std::vector<Complex> eri(static_cast<std::size_t>(n) * n * n * n);
    for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q)
            for (int a = 0; a < n; ++a)
                for (int s = 0; s < n; ++s)
                    eri[((p * n + q) * n + a) * n + s] = l(p, q) * l(a, s);
    const Eigen::MatrixXcd hd = spinor_reference::dense_hamiltonian(strings->masks(), n, h, eri);
    Action exact(basis, transform, Sigma(strings, h, eri), 1024 * 1024);
    Action ri(basis, transform, Sigma(strings, h, std::vector<Eigen::MatrixXcd>{l}), 1024 * 1024);
    const Eigen::MatrixXcd v = spinor_reference::deterministic_complex(selected, 2);
    const Eigen::MatrixXcd expected_h = b.adjoint() * t.adjoint() * hd * t * b * v;
    const Eigen::MatrixXcd expected_s = b.adjoint() * t.adjoint() * t * b * v;
    const auto images = exact.apply(v);
    close(images.hamiltonian, expected_h);
    close(images.overlap, expected_s);
    close(ri.apply(v).hamiltonian, expected_h);
    Eigen::SparseMatrix<Complex> full_mapping(d, d);
    full_mapping.setIdentity();
    Action full(Basis(strings, full_mapping), transform, Sigma(strings, h, eri), 1024 * 1024);
    const auto full_images = full.apply(x);
    close(full_images.hamiltonian, t.adjoint() * hd * t * x);
    close(full_images.overlap, t.adjoint() * t * x);
    const Eigen::MatrixXcd image = t * b * v;
    close(v.adjoint() * images.hamiltonian, image.adjoint() * hd * image);
    close(v.adjoint() * images.overlap, image.adjoint() * image);
    const Eigen::MatrixXcd gram = r.adjoint() * r;
    close(expected_s, b.adjoint() * spinor_reference::wedge_matrix(gram, strings->masks()) * b * v);
    Action bounded(basis, transform, Sigma(strings, h, eri), 1);
    rejects([&] { bounded.apply(v); });
    if (d > selected && (t * b).bottomRows(d - selected).norm() == 0.0)
        throw std::runtime_error("fixture does not probe exterior image outside selection");
}

void test_frame() {
    Eigen::MatrixXd s = Eigen::MatrixXd::Identity(3, 3);
    s(0, 1) = s(1, 0) = 0.2;
    Eigen::MatrixXcd b = spinor_reference::deterministic_complex(6, 4);
    b.topRows(4) += Eigen::MatrixXcd::Identity(4, 4);
    Frame frame(b, s);
    close(frame.coefficients() * frame.upper(), b);
    const auto& c = frame.coefficients();
    close(c.topRows(3).adjoint() * s * c.topRows(3) +
              c.bottomRows(3).adjoint() * s * c.bottomRows(3),
          Eigen::MatrixXcd::Identity(4, 4));
    b.col(1) = b.col(0);
    rejects([&] { Frame bad(b, s); });
    // Orthogonal occupied subsets can have zero overlap in a valid full frame.
    Frame identity(Eigen::MatrixXcd::Identity(6, 6), s);
    if (identity.gram()(0, 3) != Complex(0.0))
        throw std::runtime_error("zero occupied overlap fixture failed");
}
} // namespace

int main() {
    try {
        test_frame();
        test_projection(4, 2);
        test_projection(6, 3);
        test_projection(4, 0);
        test_projection(4, 4);
        std::cout << "spinor frame, exterior adjoint, selected exact/RI projection: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
