"""ENERGY-only exact SU(2) H/S with a prepared, immutable MPS file bank.

Workers share files, not mutable Block2 drivers/tensors/environments. This
module deliberately does not own RDM construction or a gradient backend.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import math
import multiprocessing
import os
import shutil
import tempfile
import time

_CONTEXT = None


def use_shared_hs(root, worker_count):
    """Keep gradient and serial execution on their existing code paths."""
    return root is None and worker_count > 1


def selected_names(names, tags):
    """The original exact file predicate, with set lookup for ordinary tags."""
    tags = list(tags)
    if any("." in tag for tag in tags):
        return [name for name in names if any(
            name == f"{tag}-mps_info.bin" or f".{tag}." in name for tag in tags)]
    tag_set = set(tags)
    info_names = {f"{tag}-mps_info.bin" for tag in tags}
    return [name for name in names if name in info_names or any(
        part in tag_set for part in name.split(".")[1:-1])]


def fingerprints(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).iterdir()) if path.is_file()}


def validate_manifest(manifest):
    for name in ("norb", "nelec", "bond", "hs_workers", "hs_threads"):
        if not isinstance(manifest[name], int) or manifest[name] < 1:
            raise ValueError(f"invalid shared H/S {name}")
    tags = manifest["state_tags"]
    if not tags or len(tags) != len(set(tags)) or any(
            not tag or "/" in tag or "\\" in tag for tag in tags):
        raise ValueError("shared H/S requires unique, local MPS tags")
    if manifest["symmetry"] != "SU2":
        raise ValueError("shared H/S requires SU(2)")


def _prepare(manifest, source, bank):
    import pair_core as core
    started = time.perf_counter()
    core.configure_phase_environment(1)
    names = selected_names(os.listdir(source), manifest["state_tags"])
    for name in names:
        path = Path(source) / name
        if path.is_file():
            shutil.copyfile(path, Path(bank) / name)
    driver = core.DMRGDriver(
        scratch=bank, symm_type=core.SymmetryTypes.SU2, n_threads=1,
        stack_mem=core.stack_memory_bytes(manifest["norb"], manifest["bond"]),
        clean_scratch=True)
    driver.initialize_system(n_sites=manifest["norb"], n_elec=manifest["nelec"],
                             spin=0, orb_sym=[0] * manifest["norb"])
    for tag in manifest["state_tags"]:
        if not (Path(bank) / f"{tag}-mps_info.bin").is_file():
            raise RuntimeError(f"shared H/S cannot find persisted MPS tag {tag}")
        state = driver.load_mps(tag)
        driver.align_mps_center(state, manifest["norb"] - state.dot)
        state, _ = driver.adjust_mps(state, dot=1)
        if state.dot != 1 or state.center != manifest["norb"] - 1:
            raise RuntimeError("shared H/S common-center preparation failed")
        state.save_data()
        state.info.save_data(str(Path(bank) / f"{tag}-mps_info.bin"))
    return time.perf_counter() - started


def _initialize(manifest, bank, scratch_parent, h1, g2):
    import pair_core as core
    global _CONTEXT
    threads = manifest["hs_threads"]
    core.configure_phase_environment(threads)
    scratch = tempfile.mkdtemp(prefix="hs-worker-", dir=scratch_parent)
    driver = core.DMRGDriver(
        scratch=scratch, symm_type=core.SymmetryTypes.SU2, n_threads=threads,
        stack_mem=core.stack_memory_bytes(manifest["norb"], manifest["bond"]),
        clean_scratch=True)
    driver.initialize_system(n_sites=manifest["norb"], n_elec=manifest["nelec"],
                             spin=0, orb_sym=[0] * manifest["norb"])
    h_mpo = core.get_hamiltonian_mpo(driver, h1, g2, log=False)
    identity = driver.get_identity_mpo()
    driver.frame.mps_dir = bank
    driver.frame.prefix_can_write = False
    states = []
    bw = driver.bw
    for tag in manifest["state_tags"]:
        info = bw.brs.MPSInfo(0)
        info.load_data(str(Path(bank) / f"{tag}-mps_info.bin"))
        if info.tag != tag:
            raise RuntimeError("shared H/S state tag mismatch")
        info.load_mutable()
        info.bond_dim = max(info.bond_dim, info.get_max_bond_dimension())
        state = bw.bs.MPS(info)
        state.load_data()
        if state.dot != 1 or state.center != manifest["norb"] - 1:
            raise RuntimeError("shared H/S reader received an incompatible center")
        state.load_mutable()
        states.append(state)
    _CONTEXT = driver, identity, h_mpo, states


def readonly_expectation(driver, bra, mpo, ket):
    if driver.frame.prefix_can_write or bra.dot != 1 or ket.dot != 1 or bra.center != ket.center:
        raise RuntimeError("shared H/S read-only layout contract failed")
    bw = driver.bw
    bond = max(bra.info.bond_dim, ket.info.bond_dim)
    environment = bw.bs.MovingEnvironment(mpo, bra, ket, "EXPT")
    environment.delayed_contraction = bw.b.OpNamesSet.normal_ops()
    environment.cached_contraction = False
    environment.fused_contraction_rotation = True
    environment.save_environments = False
    environment.init_environments(False)
    expect = bw.bs.Expect(environment, bond, bond)
    expect.store_bra_spectra = expect.store_ket_spectra = False
    expect.iprint = 0
    return float(expect.solve(False, ket.center != 0))


def _batch(pairs):
    if _CONTEXT is None:
        raise RuntimeError("shared H/S worker was not initialized")
    driver, identity, h_mpo, states = _CONTEXT
    rows = []
    for order, i, j in pairs:
        started = time.perf_counter()
        overlap = readonly_expectation(driver, states[i], identity, states[j])
        overlap_seconds = time.perf_counter() - started
        started = time.perf_counter()
        hamiltonian = readonly_expectation(driver, states[i], h_mpo, states[j])
        rows.append((order, i, j, overlap, hamiltonian, overlap_seconds,
                     time.perf_counter() - started, os.getpid()))
    return rows


def contract(states, source_scratch, scratch_parent, symmetry, norb, nelec,
             bond, h1, g2, pairs, workers, threads, state_tags=None):
    """Match pair_core.contract_parallel_hs; never mutate the source bank."""
    import pair_core as core
    tags = core.persist_parallel_states(states, source_scratch) if states is not None else state_tags
    manifest = dict(symmetry=symmetry, norb=norb, nelec=nelec, bond=bond,
                    state_tags=list(tags or []), hs_workers=workers, hs_threads=threads)
    validate_manifest(manifest)
    started = time.perf_counter()
    scratch = Path(tempfile.mkdtemp(prefix="tnvb-shared-hs-", dir=scratch_parent))
    bank = scratch / "bank"
    bank.mkdir()
    context = multiprocessing.get_context("spawn")
    try:
        # Preparing in a child avoids replacing a live construction driver's
        # process-global Block2 frame when serial state construction was used.
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            prep_seconds = pool.submit(_prepare, manifest, str(source_scratch), str(bank)).result()
        before = fingerprints(bank)
        core.configure_phase_environment(threads)
        size = max(1, math.ceil(len(pairs) / (8 * workers)))
        chunks = [pairs[i:i + size] for i in range(0, len(pairs), size)]
        rows = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                initializer=_initialize,
                initargs=(manifest, str(bank), str(scratch), h1, g2)) as pool:
            futures = [pool.submit(_batch, chunk) for chunk in chunks]
            for future in as_completed(futures):
                rows.extend(future.result())
        if before != fingerprints(bank):
            raise RuntimeError("shared H/S bank changed during read-only contraction")
        rows.sort(key=lambda row: row[0])
        if [row[0] for row in rows] != sorted(pair[0] for pair in pairs):
            raise RuntimeError("shared H/S returned missing or duplicate pair indices")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    core.log_summary(f"[TNVB-HS-BACKEND] shared-readonly prep={prep_seconds:.3f}s "
                     f"total={time.perf_counter() - started:.3f}s bank=verified")
    return rows
