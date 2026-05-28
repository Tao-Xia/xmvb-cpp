import numpy as np

np.set_printoptions(precision=4, suppress=True)

def two_pass_mgs(B, tol=1e-12):
    basis = []
    for j in range(B.shape[1]):
        v = B[:, j].astype(float).copy()
        if np.linalg.norm(v) <= tol:
            continue

        for _ in range(2):
            for u in basis:
                v -= np.dot(u, v) * u

        n = np.linalg.norm(v)
        if n <= tol:
            continue

        v /= n
        pivot = np.argmax(np.abs(v))
        if v[pivot] < 0:
            v = -v
        basis.append(v)

    return np.column_stack(basis) if basis else np.zeros((B.shape[0], 0))

def euclidean_complement(A, n_virt, tol=1e-12):
    # toy version: only for understanding; real code uses S-metric complement
    Q, _ = np.linalg.qr(A, mode="reduced")
    basis = []
    for e in np.eye(A.shape[0]).T:
        v = e - Q @ (Q.T @ e)
        for u in basis:
            v -= np.dot(u, v) * u
        n = np.linalg.norm(v)
        if n > tol:
            basis.append(v / n)
        if len(basis) == n_virt:
            break
    return np.column_stack(basis) if basis else np.zeros((A.shape[0], 0))

def build_B_p(block_occ_raw, block_virt, support_rows, p, n_inactive):
    n_active = block_occ_raw.shape[1] - n_inactive

    local_occ = block_occ_raw[support_rows, :]
    local_virt = block_virt[support_rows, :]

    cols = []
    if p < n_inactive:
        # inactive orbital: active occupied directions + virtual directions
        if n_active > 0:
            cols.append(local_occ[:, n_inactive:])
    else:
        # active orbital: inactive directions + other active directions + virtual directions
        active_idx = p - n_inactive
        cols.append(local_occ[:, :n_inactive])

        other_active = [j for j in range(n_active) if j != active_idx]
        if other_active:
            cols.append(local_occ[:, n_inactive + np.array(other_active)])

    if local_virt.size > 0:
        cols.append(local_virt)

    return np.column_stack([c for c in cols if c.size > 0])

# One block: [T_i, T_a]
T_i = np.array([
    [0.8, 0.0],
    [0.1, 0.7],
    [0.0, 0.2],
    [0.4, 0.0],
    [0.0, 0.1],
], dtype=float)

T_a = np.array([
    [0.0, 0.1],
    [0.3, 0.0],
    [0.5, 0.2],
    [0.0, 0.4],
    [0.2, 0.0],
], dtype=float)

block_occ_raw = np.hstack([T_i, T_a])   # this is the local [T_i, T_a]
block_virt = euclidean_complement(block_occ_raw, n_virt=2)

support_rows = np.array([0, 1, 3, 4])    # AO support of orbital p
p = 2                                    # choose an active orbital
n_inactive = T_i.shape[1]

B_p = build_B_p(block_occ_raw, block_virt, support_rows, p, n_inactive)
U_p = two_pass_mgs(B_p)

print("block_occ_raw =\n", block_occ_raw)
print("block_virt =\n", block_virt)
print("B_p =\n", B_p)
print("U_p^T U_p =\n", U_p.T @ U_p)