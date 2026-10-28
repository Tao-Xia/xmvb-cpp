#pragma once
#include <bit>
#include <cstdint>
#ifdef __CUDACC__
#define XMVB_CANONICAL __device__ __forceinline__
#else
#define XMVB_CANONICAL inline
#endif
namespace xmvb::mcscf::canonical {
using Mask = std::uint64_t;
XMVB_CANONICAL int population(Mask x) {
#ifdef __CUDA_ARCH__
    return __popcll(x);
#else
    return std::popcount(x);
#endif
}
XMVB_CANONICAL int trailing(Mask x) {
#ifdef __CUDA_ARCH__
    return __ffsll(x) - 1;
#else
    return std::countr_zero(x);
#endif
}
XMVB_CANONICAL double single_sign(Mask source, int removed, int created) {
    const int low = removed < created ? removed : created,
              high = removed < created ? created : removed;
    const Mask between = ((Mask{1} << high) - 1) ^ ((Mask{1} << (low + 1)) - 1);
    return population(source & between) % 2 ? -1.0 : 1.0;
}
XMVB_CANONICAL double double_sign(Mask source, Mask removed, Mask created) {
    const int sites[4]{trailing(removed),
                       trailing(removed & (removed - 1)),
                       trailing(created & (created - 1)),
                       trailing(created)};
    int sign = 1;
    for (int i : sites) {
        if (population(source & ((Mask{1} << i) - 1)) % 2)
            sign = -sign;
        source ^= Mask{1} << i;
    }
    return sign;
}
/** @brief Shared host/device canonical Slater--Condon sign and term algebra.
 * Pair interchange is allowed; symmetry within an integral pair is not assumed.
 */
template <class One, class Two>
XMVB_CANONICAL void terms(Mask ta, Mask tb, Mask sa, Mask sb, One&& one, Two&& two) {
    const Mask ra = sa & ~ta, rb = sb & ~tb, ca = ta & ~sa, cb = tb & ~sb;
    const int da = population(ra), db = population(rb);
    if (da + db == 0) {
        for (int si = 0; si < 2; ++si) {
            const Mask spin = si ? sb : sa;
            for (Mask im = spin; im; im &= im - 1) {
                const int i = trailing(im);
                one(i, i, 1.0);
                for (int oi = 0; oi < 2; ++oi)
                    for (Mask jm = oi ? sb : sa; jm; jm &= jm - 1) {
                        const int j = trailing(jm);
                        two(i, i, j, j, 0.5);
                    }
                for (Mask jm = spin; jm; jm &= jm - 1) {
                    const int j = trailing(jm);
                    two(i, j, j, i, -0.5);
                }
            }
        }
    } else if (da + db == 1) {
        const Mask occupied = da ? sa : sb;
        const int p = trailing(da ? ca : cb), q = trailing(da ? ra : rb);
        const double sign = single_sign(occupied, q, p);
        one(p, q, sign);
        for (int si = 0; si < 2; ++si)
            for (Mask jm = si ? sb : sa; jm; jm &= jm - 1) {
                const int j = trailing(jm);
                two(p, q, j, j, sign);
            }
        for (Mask jm = occupied; jm; jm &= jm - 1) {
            const int j = trailing(jm);
            two(p, j, j, q, -sign);
        }
    } else if (da == 1 && db == 1) {
        const int p = trailing(ca), q = trailing(ra), r = trailing(cb), s = trailing(rb);
        two(p, q, r, s, single_sign(sa, q, p) * single_sign(sb, s, r));
    } else if (da + db == 2) {
        const Mask removed = da ? ra : rb, created = da ? ca : cb;
        const int p = trailing(created), r = trailing(created & (created - 1));
        const int q = trailing(removed), s = trailing(removed & (removed - 1));
        const double sign = double_sign(da ? sa : sb, removed, created);
        two(p, q, r, s, sign);
        two(p, s, r, q, -sign);
    }
}
/** @brief Exact covalent source neighbors of a target with at most two ionic pairs.
 * Requires complete complement-paired original support, not merely a COV label.
 */
template <class Consumer>
XMVB_CANONICAL void covalent_sources(Mask a, Mask b, Mask full, Consumer&& add) {
    const Mask d = a & b, e = full ^ (a | b);
    const int k = population(d);
    if (k == 0) {
        add(a);
        for (Mask p = a; p; p &= p - 1)
            for (Mask q = b; q; q &= q - 1)
                add(a ^ (p & -p) ^ (q & -q));
    } else if (k == 1) {
        add(a);
        add(a ^ d ^ e);
        for (Mask rest = full ^ (d | e); rest; rest &= rest - 1) {
            const Mask bit = rest & -rest;
            add(a & bit ? a ^ e ^ bit : a ^ d ^ bit);
        }
    } else if (k == 2) {
        const Mask d0 = d & -d, d1 = d ^ d0, e0 = e & -e, e1 = e ^ e0;
        add(a);
        add(a ^ d0 ^ e0);
        add(a ^ d0 ^ e1);
        add(a ^ d1 ^ e1);
        add(a ^ d1 ^ e0);
        add(a ^ d ^ e);
    }
}
} // namespace xmvb::mcscf::canonical
#undef XMVB_CANONICAL
