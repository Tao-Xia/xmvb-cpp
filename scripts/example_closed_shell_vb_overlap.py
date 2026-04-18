#!/usr/bin/env python3
import itertools, math, numpy as np

def parse(text):
    return [(int(a) - 1, int(b) - 1) for a, b in (t.split("-") for t in text.split())]

def sign_sort(xs):
    xs, s = list(xs), 1
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            if xs[i] > xs[j]:
                xs[i], xs[j], s = xs[j], xs[i], -s
    return tuple(xs), s

def expand(pairs):
    terms = {((), ()): 1.0}
    for i, j in pairs:
        nxt = {}
        for (a, b), c in terms.items():
            k = (a + (i,), b + (j,)); nxt[k] = nxt.get(k, 0.0) + c
            if i != j:
                k = (a + (j,), b + (i,)); nxt[k] = nxt.get(k, 0.0) - c
        terms = nxt
    out = {}
    for (a, b), c in terms.items():
        a, sa = sign_sort(a); b, sb = sign_sort(b)
        out[(a, b)] = out.get((a, b), 0.0) + c * sa * sb
    return {k: v for k, v in out.items() if abs(v) > 1.0e-12}

def det_overlap(left_pairs, right_pairs, S):
    if not left_pairs: return 1.0
    left, right, n, total = expand(left_pairs), expand(right_pairs), len(left_pairs), 0.0
    for (la, lb), lc in left.items():
        for (ra, rb), rc in right.items():
            A = np.array([[S[la[i], ra[j]] for j in range(n)] for i in range(n)], float)
            B = np.array([[S[lb[i], rb[j]] for j in range(n)] for i in range(n)], float)
            total += lc * rc * np.linalg.det(A) * np.linalg.det(B)
    return total

def pf(M):
    M, n, v = M.astype(float).copy(), M.shape[0], 1.0
    if n % 2: return 0.0
    for i in range(0, n - 1, 2):
        j = max(range(i + 1, n), key=lambda k: abs(M[i, k]))
        if abs(M[i, j]) <= 1.0e-14: return 0.0
        if j != i + 1:
            M[[i + 1, j], :] = M[[j, i + 1], :]
            M[:, [i + 1, j]] = M[:, [j, i + 1]]
            v = -v
        p = M[i, i + 1]; v *= p
        for r in range(i + 2, n):
            for c in range(r + 1, n):
                M[r, c] -= (M[i, r] * M[i + 1, c] - M[i, c] * M[i + 1, r]) / p
                M[c, r] = -M[r, c]
    return v

def pair_matrix(m, i, j):
    F, a, b = np.zeros((2 * m, 2 * m)), i, m + i
    if i == j: F[a, b], F[b, a] = 1.0, -1.0; return F
    c, d = j, m + j
    F[a, d], F[d, a], F[c, b], F[b, c] = 1.0, -1.0, -1.0, 1.0
    return F

def compute_raw_vb_pfaffian_overlap(left_pairs, right_pairs, S):
    if len(left_pairs) != len(right_pairs): raise ValueError("pair counts must match")
    if not left_pairs: return 1.0
    m, n = S.shape[0], len(left_pairs)
    Lf = [pair_matrix(m, i, j) for i, j in left_pairs]
    Rf = [pair_matrix(m, i, j) for i, j in right_pairs]
    Sigma = np.zeros((2 * m, 2 * m)); Sigma[:m, :m] = S; Sigma[m:, m:] = S
    total = 0.0
    for ass in itertools.product((-1.0, 1.0), repeat=2 * n):
        L = np.zeros((2 * m, 2 * m)); R = np.zeros((2 * m, 2 * m))
        for p in range(n): L += ass[p] * Lf[p]
        for p in range(n): R += ass[n + p] * Rf[p]
        B = np.zeros((4 * m, 4 * m))
        B[:2 * m, :2 * m], B[:2 * m, 2 * m:] = L, Sigma
        B[2 * m:, :2 * m], B[2 * m:, 2 * m:] = -Sigma.T, R
        total += np.prod(ass) * pf(B)
    return total / (2.0 ** (2 * n))

def main():
    left_text, right_text = "1-2 3-4", "1-2 3-4"
    left, right = parse(left_text), parse(right_text)
    m = max(max(i, j) for i, j in left + right) + 1
    S = np.eye(m)
    s11d, s22d, s12d = det_overlap(left, left, S), det_overlap(right, right, S), det_overlap(left, right, S)
    s11p, s22p, s12p = compute_raw_vb_pfaffian_overlap(left, left, S), compute_raw_vb_pfaffian_overlap(right, right, S), compute_raw_vb_pfaffian_overlap(left, right, S)
    print("left =", left_text); print("right =", right_text); print("S =\n", S)
    print("det:", s11d, s22d, s12d); print("pf :", s11p, s22p, s12p)
    print("abs:", abs(s11d - s11p), abs(s22d - s22p), abs(s12d - s12p))
    nd, npf = math.sqrt(max(1e-30, s11d * s22d)), math.sqrt(max(1e-30, s11p * s22p))
    print("norm:", s12d / nd, s12p / npf)

if __name__ == "__main__":
    main()
