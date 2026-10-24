#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace xmvb::mcscf::diagnostics {

/** @brief Full CIAH diagnostics, including explicit spectral/model audits. */
inline bool ciah_trace_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("XMVB_CIAH_TRACE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

/** @brief Existing events only; scalar mode leaves the native numerical path unchanged. */
inline bool ciah_event_trace_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("XMVB_CIAH_TRACE");
        return value && (std::strcmp(value, "1") == 0 || std::strcmp(value, "scalar") == 0);
    }();
    return enabled;
}

/** @brief Expensive nonlinear candidate audit; never a production policy. */
inline bool ciah_candidate_audit_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("XMVB_CIAH_CANDIDATE_AUDIT");
        return ciah_trace_enabled() && value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

/** @brief Identifies an accepted-point solve and its radius retry. */
struct CiahTraceContext {
    int keyframe = 0;
    int attempt = 0;
    const char* source = "initial";
};

inline thread_local CiahTraceContext ciah_trace_context;

/** @brief Writes existing scalar diagnostics without retaining vectors. */
template <class... Args>
inline void ciah_trace(const char* event, const char* format, Args... args) {
    if (!ciah_event_trace_enabled())
        return;
    std::fprintf(stderr,
                 "CIAH_DIAG,%d,%d,%s,",
                 ciah_trace_context.keyframe,
                 ciah_trace_context.attempt,
                 event);
    std::fprintf(stderr, format, args...);
    std::fputc('\n', stderr);
}

} // namespace xmvb::mcscf::diagnostics
