#include "core/memory.hpp"
#include "core/workspace.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace xmvb::memory {
namespace {

std::optional<std::size_t> number(std::string_view text) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        return {};
    return value;
}

std::string token(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::string value;
    file >> value;
    return value;
}

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("memory resource size overflow");
    return a * b;
}

std::optional<std::size_t> field(const std::filesystem::path& path, std::string_view name) {
    std::ifstream file(path);
    std::string line, key;
    std::size_t value;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        if ((row >> key >> value) && key == name)
            return multiply(value, 1024);
    }
    return {};
}

// mountinfo escapes whitespace and backslashes using octal codes.
std::string unescape(std::string value) {
    std::string result;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 3 < value.size() && value[i + 1] >= '0' &&
            value[i + 1] <= '7' && value[i + 2] >= '0' && value[i + 2] <= '7' &&
            value[i + 3] >= '0' && value[i + 3] <= '7') {
            result += static_cast<char>((value[i + 1] - '0') * 64 + (value[i + 2] - '0') * 8 +
                                        value[i + 3] - '0');
            i += 3;
        } else
            result += value[i];
    }
    return result;
}

bool contains(std::string_view list, std::string_view entry) {
    return ("," + std::string(list) + ",").find("," + std::string(entry) + ",") !=
           std::string::npos;
}

void cgroups(Snapshot& snapshot, const std::filesystem::path& proc, std::size_t physical) {
    std::ifstream groups(proc / "self/cgroup"), mounts(proc / "self/mountinfo");
    std::string line, v1, v2;
    while (std::getline(groups, line)) {
        const auto first = line.find(':'), second = line.find(':', first + 1);
        if (first == line.npos || second == line.npos)
            continue;
        const auto controllers = line.substr(first + 1, second - first - 1);
        if (controllers.empty())
            v2 = line.substr(second + 1);
        if (contains(controllers, "memory"))
            v1 = line.substr(second + 1);
    }
    while (std::getline(mounts, line)) {
        const auto separator = line.find(" - ");
        if (separator == line.npos)
            continue;
        std::istringstream prefix(line.substr(0, separator)), suffix(line.substr(separator + 3));
        std::string id, parent, device, root, mount, options, type, source, super;
        if (!(prefix >> id >> parent >> device >> root >> mount >> options) ||
            !(suffix >> type >> source >> super))
            continue;
        const bool unified = type == "cgroup2";
        if (!unified && !(type == "cgroup" && contains(super, "memory")))
            continue;
        const auto& group = unified ? v2 : v1;
        if (group.empty())
            continue;
        const auto base = std::filesystem::path(unescape(mount)).lexically_normal();
        const auto mount_root = std::filesystem::path(unescape(root)).lexically_normal();
        const auto relative =
            std::filesystem::path(group).lexically_normal().lexically_relative(mount_root);
        if (relative.empty() || *relative.begin() == "..")
            continue;
        auto current = (base / relative).lexically_normal();
        for (;;) {
            const auto limit =
                number(token(current / (unified ? "memory.max" : "memory.limit_in_bytes")));
            const auto used =
                number(token(current / (unified ? "memory.current" : "memory.usage_in_bytes")));
            // v1's near-SIZE_MAX unlimited sentinel is not a resource allocation.
            if (limit && used && (unified || !physical || *limit <= physical))
                snapshot.constraints.push_back({"cgroup:" + current.string(), *limit, *used});
            if (current == base)
                break;
            current = current.parent_path();
        }
    }
}

std::optional<std::size_t> environment(const char* name) {
    const auto value = std::getenv(name);
    return value ? number(value) : std::nullopt;
}

std::optional<std::size_t> upper_bound;
bool cli_policy = false, ci_prepared = false;

void report(const Budget& budget, std::string_view stage) {
    // Do not change the application stream's formatting state.
    std::ostringstream row;
    constexpr double gib = 1024.0 * 1024 * 1024;
    row << std::fixed << std::setprecision(3) << "CPU workspace (" << stage
        << "): " << budget.bytes / gib << " GiB; source=" << budget.source
        << "; limit=" << budget.limit / gib << " GiB; usage=" << budget.used / gib
        << " GiB; pending solver=" << budget.pending / gib << " GiB";
    if (upper_bound)
        row << "; requested upper=" << *upper_bound / gib << " GiB";
    std::cout << row.str() << '\n';
}

} // namespace

Snapshot inspect(const std::filesystem::path& proc) {
    Snapshot result;
    result.rss = field(proc / "self/status", "VmRSS:").value_or(0);
    const auto total = field(proc / "meminfo", "MemTotal:").value_or(0);
    const auto available = field(proc / "meminfo", "MemAvailable:");
    if (available)
        result.constraints.push_back({"MemAvailable", *available, 0});
    cgroups(result, proc, total);
    // Fixture reads are independent of the runner's Slurm environment.
    if (proc == std::filesystem::path("/proc")) {
        auto mib = environment("SLURM_MEM_PER_NODE");
        if (!mib || !*mib) {
            const auto per_cpu = environment("SLURM_MEM_PER_CPU");
            const auto cpus = environment("SLURM_CPUS_ON_NODE");
            if (per_cpu && cpus)
                mib = multiply(*per_cpu, *cpus);
        }
        if (mib && *mib)
            result.constraints.push_back(
                {"Slurm allocation (process RSS usage)", multiply(*mib, 1024 * 1024), result.rss});
    }
    return result;
}

Budget plan(const Snapshot& snapshot, std::size_t pending, std::optional<std::size_t> upper) {
    Budget result;
    auto free = std::numeric_limits<std::size_t>::max();
    for (const auto& constraint : snapshot.constraints) {
        const auto headroom = constraint.limit - std::min(constraint.limit, constraint.used);
        if (headroom < free) {
            free = headroom;
            result.source = constraint.source;
            result.limit = constraint.limit;
            result.used = constraint.used;
        }
    }
    if (result.source.empty()) {
        if (!upper)
            throw std::runtime_error(
                "cannot discover CPU memory resources; set --ci-workspace-mib explicitly");
        free = *upper;
        result.source = "explicit upper (resources unavailable)";
        result.limit = free;
    }
    result.pending = pending;
    result.bytes = free - std::min(free, pending);
    if (upper)
        result.bytes = std::min(result.bytes, *upper);
    if (!result.bytes)
        throw std::runtime_error(
            "CPU memory headroom is exhausted before numerical workspace allocation");
    return result;
}

std::size_t default_workspace() {
    static const auto bytes = plan(inspect()).bytes;
    return bytes;
}

void configure(std::optional<std::size_t> upper) {
    if (upper && !*upper)
        throw std::invalid_argument("workspace upper bound must be positive");
    upper_bound = upper;
    cli_policy = true;
    ci_prepared = false;
    const auto budget = plan(inspect(), 0, upper);
    workspace::set_limit(budget.bytes);
    report(budget, "preparation");
}

void prepare_ci(std::size_t pending) {
    if (!cli_policy || ci_prepared)
        return;
    if (workspace::has_scope())
        throw std::logic_error("CPU memory planning requires a phase boundary");
    // Release idle scratch before observing RSS. Live retained data stays charged.
    workspace::reclaim_idle();
    const auto budget = plan(inspect(), pending, upper_bound);
    workspace::set_limit(budget.bytes);
    ci_prepared = true;
    report(budget, "first CI");
}

} // namespace xmvb::memory
