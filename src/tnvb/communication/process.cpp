#include "tnvb/communication/process.hpp"
#include "tnvb/config.hpp"
#include "core/index.hpp"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <spawn.h>
#include <stdexcept>
#include <system_error>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace xmvb::tnvb {
namespace {

std::filesystem::path unique_directory(const std::filesystem::path& parent) {
    std::string pattern = (parent / "xmvb-cpp-tnvb-XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) {
        throw std::system_error(errno, std::generic_category(), "TNVB scratch creation");
    }
    return pattern;
}

template <class T> void write_values(std::ostream& stream, const T* data, std::size_t count) {
    const auto bytes = core::checked_multiply(count, sizeof(T));
    stream.write(reinterpret_cast<const char*>(data), core::checked_index<std::streamsize>(bytes));
    if (!stream)
        throw std::runtime_error("TNVB request write failed");
}

template <class T> void read_values(std::istream& stream, T* data, std::size_t count) {
    const auto bytes = core::checked_multiply(count, sizeof(T));
    stream.read(reinterpret_cast<char*>(data), core::checked_index<std::streamsize>(bytes));
    if (!stream)
        throw std::runtime_error("truncated TNVB worker response");
}

std::filesystem::path worker_path() {
    // Prefer the matched installed worker relative to the executable; builds
    // use their own staged copy. Never fall back to xmvb4.0 source files.
    std::array<char, 4096> executable{};
    const ssize_t length = ::readlink("/proc/self/exe", executable.data(), executable.size() - 1);
    if (length > 0) {
        const auto candidate = std::filesystem::path(std::string(executable.data(), length))
                                   .parent_path()
                                   .parent_path() /
                               "libexec/xmvb-cpp/tnvb/evaluate.py";
        if (std::filesystem::exists(candidate))
            return candidate;
    }
    if (std::filesystem::exists(XMVB_TNVB_WORKER))
        return XMVB_TNVB_WORKER;
    throw std::runtime_error("matched independent TNVB worker is missing");
}

void execute_worker(const std::filesystem::path& request,
                    const std::filesystem::path& response,
                    const std::filesystem::path& cache,
                    const std::string& handoff_option,
                    const std::filesystem::path& handoff_directory) {
    std::map<std::string, std::string> environment;
    for (char** item = environ; *item != nullptr; ++item) {
        const std::string entry(*item);
        const auto equals = entry.find('=');
        if (equals != std::string::npos) {
            environment.emplace(entry.substr(0, equals), entry.substr(equals + 1));
        }
    }
    environment["PYTHONPATH"] =
        std::string(XMVB_TNVB_BLOCK2_MODULE) + ":" + XMVB_TNVB_BLOCK2_SOURCE;
    const auto previous_preload = environment.find("LD_PRELOAD");
    const std::string inherited_preload =
        previous_preload == environment.end() ? std::string{} : previous_preload->second;
    environment["LD_PRELOAD"] =
        std::string(XMVB_TNVB_LIBRARY_DIR) + "/libstdc++.so.6:" + XMVB_TNVB_LIBRARY_DIR +
        "/libgomp.so.1" + (inherited_preload.empty() ? std::string{} : ":" + inherited_preload);
    environment["XMVB_BLOCK2_WARM_CACHE"] = cache.string();
    environment["XMVB_BLOCK2_SCRATCH_PARENT"] = request.parent_path().string();
    environment["XMVB_BLOCK2_RDM_CHECK_TOL"] = "1e-7";
    environment["OPENBLAS_NUM_THREADS"] = "1";
    environment["OMP_DYNAMIC"] = "FALSE";
    environment["OMP_MAX_ACTIVE_LEVELS"] = "1";
    std::vector<std::string> entries;
    entries.reserve(environment.size());
    for (const auto& item : environment)
        entries.push_back(item.first + "=" + item.second);
    std::vector<char*> envp;
    envp.reserve(entries.size() + 1);
    for (auto& item : entries)
        envp.push_back(item.data());
    envp.push_back(nullptr);
    std::string python = XMVB_TNVB_PYTHON;
    std::string worker = worker_path().string();
    std::string in = request.string(), out = response.string();
    std::string handoff = handoff_directory.string();
    std::string option = handoff_option;
    std::vector<char*> argv{python.data(), worker.data(), in.data(), out.data()};
    if (!option.empty()) {
        argv.push_back(option.data());
        argv.push_back(handoff.data());
    }
    argv.push_back(nullptr);
    // Flush parent reports before the child writes to the inherited stdout.
    // Otherwise redirected logs can splice worker summaries into an orbital row.
    std::cout.flush();
    std::cerr.flush();
    pid_t child = -1;
    const int error =
        ::posix_spawn(&child, python.c_str(), nullptr, nullptr, argv.data(), envp.data());
    if (error != 0) {
        throw std::system_error(error, std::generic_category(), "TNVB worker launch");
    }
    int status = 0;
    while (::waitpid(child, &status, 0) == -1) {
        if (errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "TNVB worker wait");
        }
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw std::runtime_error("TNVB worker failed (wait status " + std::to_string(status) + ")");
    }
}

} // namespace

WorkerProcess::WorkerProcess() {
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1) {
        throw std::runtime_error("TNVB binary protocol requires a little-endian host");
    }
    const char* parent = std::getenv("XMVB_BLOCK2_SCRATCH_PARENT");
    if (!(parent && *parent))
        parent = std::getenv("SLURM_TMPDIR");
    scratch_ = std::shared_ptr<const StateDirectory>(new StateDirectory(
        unique_directory(parent && *parent ? parent : std::filesystem::temp_directory_path())));
}

StateDirectory::~StateDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
}

WorkerProcess::~WorkerProcess() = default;

WorkerResponse WorkerProcess::evaluate(const vb::VbScfInput& input,
                                       const vb::PreparedActiveSpaceContext& prepared,
                                       const vb::ElectronicStateRequest& electronic_request,
                                       bool gradient,
                                       bool retain_forward,
                                       std::shared_ptr<const StateDirectory> resume) const {
    if ((retain_forward && gradient) || (resume && (!gradient || retain_forward)) ||
        (resume && resume->parent_.get() != scratch_.get())) {
        throw std::invalid_argument("invalid or foreign TNVB handoff lease");
    }
    const auto lease = std::shared_ptr<const StateDirectory>(
        new StateDirectory(unique_directory(scratch_->path()), scratch_));
    const auto& directory = lease->path();
    const auto request = resume ? resume->path() / "request.bin" : directory / "request.bin";
    const auto response = directory / "response.bin";
    const auto& topology = input.raw_structure_data;
    // Protocol v3 has explicit 32-bit dimensions; upstream counts are native-wide.
    // Reject an unrepresentable request before any narrowing, allocation or launch.
    const auto n_str = core::checked_index<std::int32_t>(topology.n_structures);
    const auto n_ao =
        core::checked_index<std::int32_t>(input.orbital_preparation_input.n_active_orbitals);
    if (n_str <= 0 || n_ao <= 0 || topology.n_total_electrons <= 0) {
        throw std::invalid_argument("TNVB raw structure dimensions must be positive");
    }
    const auto n_matrix = core::checked_multiply(std::size_t(n_ao), std::size_t(n_ao));
    const auto n_pair = core::triangular_count(std::size_t(n_ao));
    const auto n_eri = core::triangular_count(n_pair);
    if (n_str <= 0 || n_ao <= 0 ||
        topology.raw_structure_orbitals.size() !=
            core::checked_multiply(std::size_t(n_str), std::size_t(topology.n_total_electrons))) {
        throw std::invalid_argument("TNVB raw structure dimensions are inconsistent");
    }
    if (!resume) {
        const auto& frame = prepared.active_space_two_electron_result.orbital_transform;
        const auto& overlap = prepared.orbital_result.active_orbital_overlap_matrix;
        if (frame.rows() != n_ao || frame.cols() != n_ao || !frame.allFinite() ||
            (frame.diagonal().array() <= 0).any() ||
            frame.triangularView<Eigen::StrictlyLower>().toDenseMatrix().norm() >
                1e-12 * std::max(1.0, frame.norm()) ||
            overlap.size() != n_matrix) {
            throw std::invalid_argument(
                "TNVB requires orthogonal active integrals and their upper orbital frame");
        }
        const Eigen::Map<const Eigen::MatrixXd> orbital_overlap(overlap.data(), n_ao, n_ao);
        if (!orbital_overlap.allFinite() ||
            (frame.transpose() * frame - orbital_overlap).cwiseAbs().maxCoeff() >
                1e-10 * std::max(1.0, orbital_overlap.cwiseAbs().maxCoeff())) {
            throw std::invalid_argument("TNVB orbital frame does not match the active overlap");
        }
        std::vector<double> packed =
            prepared.active_space_two_electron_result.packed_active_two_electron_integrals;
        if (packed.empty() &&
            prepared.active_space_two_electron_result.representation ==
                integrals::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
            const auto& factors = prepared.active_space_two_electron_result.ri_active_pair_factors;
            if (factors.cols() != static_cast<Eigen::Index>(n_pair)) {
                throw std::invalid_argument("TNVB RI active-pair dimensions are inconsistent");
            }
            packed.resize(n_eri);
            for (std::size_t q = 0; q < n_pair; ++q) {
                for (std::size_t p = 0; p <= q; ++p) {
                    packed[q * (q + 1) / 2 + p] = factors.col(p).dot(factors.col(q));
                }
            }
        }
        if (packed.size() != n_eri)
            throw std::invalid_argument("TNVB active ERI size mismatch");
        {
            std::ofstream stream(request, std::ios::binary);
            electronic_request.accuracy.validate();
            const auto& initial = electronic_request.initial_eigenvectors;
            const bool davidson =
                electronic_request.eigensolver == vb::StructureEigensolver::Davidson;
            const bool has_guess = davidson && initial.size() != 0;
            if (has_guess &&
                (initial.rows() != n_str || initial.cols() < 1 || !initial.allFinite())) {
                throw std::invalid_argument("invalid TNVB Davidson initial coefficients");
            }
            stream.write("CPTNREQ3", 8);
            const std::array<std::int32_t, 10> header{3,
                                                      n_str,
                                                      n_ao,
                                                      topology.n_active_electrons,
                                                      topology.n_total_electrons,
                                                      input.tnvb.structure_bond,
                                                      input.tnvb.total_bond,
                                                      gradient ? 1 : 0,
                                                      davidson ? 1 : 0,
                                                      has_guess ? 1 : 0};
            write_values(stream, header.data(), header.size());
            const std::array<double, 2> accuracy{electronic_request.accuracy.energy_tolerance,
                                                 electronic_request.accuracy.gradient_tolerance};
            write_values(stream, accuracy.data(), accuracy.size());
            if (has_guess)
                write_values(stream, initial.col(0).data(), n_str);
            // RawStructureData is already structure-major, i.e. Fortran (NEL,NSTR).
            for (const int orbital : topology.raw_structure_orbitals) {
                const std::int32_t value = orbital;
                write_values(stream, &value, 1);
            }
            const auto& h = prepared.active_space_one_electron_result.h1e_act;
            if (overlap.size() != n_matrix || std::size_t(h.size()) != n_matrix) {
                throw std::invalid_argument("TNVB active matrix size mismatch");
            }
            write_values(stream, overlap.data(), overlap.size());
            write_values(stream, frame.data(), frame.size());
            write_values(stream, h.data(), h.size());
            write_values(stream, packed.data(), packed.size());
        }
    }
    const std::string option = resume ? "--resume" : retain_forward ? "--retain" : "";
    const auto handoff_directory = resume ? resume->path() / "forward" : directory / "forward";
    execute_worker(request, response, scratch_->path() / "warm", option, handoff_directory);
    WorkerResponse result;
    std::ifstream stream(response, std::ios::binary);
    std::array<char, 8> magic{};
    read_values(stream, magic.data(), magic.size());
    std::array<std::int32_t, 8> header{};
    read_values(stream, header.data(), header.size());
    const bool davidson = electronic_request.eigensolver == vb::StructureEigensolver::Davidson;
    const int n_roots = davidson ? 1 : n_str;
    if (std::memcmp(magic.data(), "CPTNRES3", 8) || header[0] != 3 || header[1] != 0 ||
        header[2] != n_str || header[3] != n_ao || header[4] != (gradient ? 1 : 0) ||
        header[5] != n_roots || header[6] != (davidson ? 1 : 0) || header[7]) {
        throw std::runtime_error("TNVB response header mismatch");
    }
    result.hamiltonian.resize(n_str, n_str);
    result.overlap.resize(n_str, n_str);
    result.energies.resize(n_roots);
    result.coefficients.resize(n_str, n_roots);
    read_values(stream, result.hamiltonian.data(), result.hamiltonian.size());
    read_values(stream, result.overlap.data(), result.overlap.size());
    read_values(stream, result.energies.data(), result.energies.size());
    read_values(stream, result.coefficients.data(), result.coefficients.size());
    std::array<double, 4> checks{};
    read_values(stream, checks.data(), checks.size());
    result.fit_energy_error = checks[0];
    result.rdm_identity_error = checks[1];
    result.hs_wall_seconds = checks[2];
    result.rdm_wall_seconds = checks[3];
    if (davidson) {
        std::array<std::int32_t, 3> counts{};
        std::array<double, 2> residuals{};
        read_values(stream, counts.data(), counts.size());
        read_values(stream, residuals.data(), residuals.size());
        if (counts[0] < 1 || counts[1] < 1 || counts[2] < 1 || counts[2] > n_str ||
            !std::isfinite(residuals[0]) || !std::isfinite(residuals[1]) || residuals[0] < 0 ||
            residuals[1] < 0) {
            throw std::runtime_error("invalid TNVB Davidson diagnostics");
        }
        result.davidson_diagnostics =
            vb::DavidsonDiagnostics{counts[0], counts[1], counts[2], residuals[0], residuals[1]};
    }
    if (gradient) {
        result.density_one.resize(n_ao, n_ao);
        result.contracted_three.resize(n_ao, n_ao);
        result.density_two_buffer.resize(core::checked_multiply(n_matrix, n_matrix));
        read_values(stream, result.density_one.data(), result.density_one.size());
        read_values(stream, result.density_two_buffer.data(), result.density_two_buffer.size());
        read_values(stream, result.contracted_three.data(), result.contracted_three.size());
    }
    if (stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("trailing TNVB response data");
    }
    if (retain_forward) {
        if (!std::filesystem::is_regular_file(directory / "forward/ready.json")) {
            throw std::runtime_error("TNVB worker did not publish its forward state");
        }
        result.retained_state = lease;
    }
    return result;
}

WorkerResponse WorkerProcess::complete_gradient(const vb::VbScfInput& input,
                                                const vb::ElectronicStateRequest& request,
                                                std::shared_ptr<const StateDirectory> state) const {
    if (!state)
        throw std::invalid_argument("missing TNVB handoff lease");
    // The retained original request contains the exact ENERGY integrals. The
    // empty preparation is never read by the resume branch above.
    const vb::PreparedActiveSpaceContext unused;
    return evaluate(input, unused, request, true, false, std::move(state));
}

} // namespace xmvb::tnvb
