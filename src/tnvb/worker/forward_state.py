"""One-shot, trial-owned ENERGY -> FULL-gradient handoff.

No live Block2 object crosses processes. Only final private MPS files, the
orthogonal integral context and the already solved CI result are retained.
The C++ lease owns the enclosing request directory, including failed trials.
"""
from pathlib import Path
import hashlib
import json
import os
import struct
import time
import numpy as np


def signature(request_path):
    data = bytearray(Path(request_path).read_bytes())
    if len(data) < 64 or data[:8] != b'CPTNREQ3':
        raise ValueError('invalid TNVB handoff request')
    header = struct.unpack('<10i',data[8:48])
    if header[0] != 3 or header[7] not in (0,1):
        raise ValueError('invalid TNVB handoff mode')
    # Only ENERGY versus GRADIENT may differ. Includes structures, all active
    # integrals, bond caps, CI solver/accuracy and any supplied initial guess.
    struct.pack_into('<i',data,36,0)
    settings = {name:os.environ.get(name,default) for name,default in (
        ('XMVB_BLOCK2_SU2_SWEEPS','4'),('XMVB_BLOCK2_SU2_TOL','1e-10'),
        ('XMVB_BLOCK2_SU2_CUTOFF','1e-14'),('XMVB_BLOCK2_SU2_SEED','20260813'),
        ('XMVB_BLOCK2_H_MPO','no-trans-conventional'))}
    h = hashlib.sha256(data)
    h.update(json.dumps(settings,sort_keys=True).encode())
    for name in ('evaluate.py','pair_core.py','forward_state.py','common_ci.py',
                 'fixed_family.py','shared_effective.py','davidson.py','hs_shared.py',
                 'rdm_full.py'):
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def inventory(directory):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError('invalid TNVB retained state file')
        stat = path.stat()
        result[path.name] = [stat.st_size,stat.st_mtime_ns]
    return result


def export_context(retain, source, tags, rinv, h1, g2):
    """Called only after construction/H/S drivers have released the bank."""
    import pair_core as core
    started=time.perf_counter()
    root=Path(retain['directory'])
    root.mkdir(exist_ok=False)
    bank=root/'mps'
    bank.mkdir()
    count=core.move_parallel_state_files(source,str(bank),tags)
    np.savez(root/'context.npz',rinv=rinv,h1=h1,g2=g2)
    retain.update(state_tags=list(tags),files=count,
                  seconds=time.perf_counter()-started)


def publish(directory, request, response, retain):
    root=Path(directory)
    if not retain.get('state_tags'):
        raise RuntimeError('TNVB handoff did not retain final MPS states')
    manifest=dict(schema=1,signature=signature(request),state_tags=retain['state_tags'],
        files=inventory(root/'mps'),context=inventory_file(root/'context.npz'),
        response=hashlib.sha256(Path(response).read_bytes()).hexdigest())
    temporary=root/'ready.tmp'
    temporary.write_text(json.dumps(manifest,sort_keys=True))
    os.replace(temporary,root/'ready.json')
    print(f'[TNVB-HANDOFF] retained structures={len(retain["state_tags"])} '
          f'files={retain["files"]} transfer=move seconds={retain["seconds"]:.3f}',flush=True)


def inventory_file(path):
    stat=path.stat()
    return [stat.st_size,stat.st_mtime_ns]


def claim(directory, request, response):
    """Validate before claiming; never silently rebuild a mismatched trial."""
    root=Path(directory)
    with (root/'ready.json').open() as stream:
        manifest=json.load(stream)
    if manifest.get('schema')!=1 or manifest.get('signature')!=signature(request):
        original=root.parent/'request.bin'
        detail=''
        if original.is_file():
            a=bytearray(original.read_bytes());b=bytearray(Path(request).read_bytes())
            if len(a)>=40 and len(b)>=40:
                struct.pack_into('<i',a,36,0);struct.pack_into('<i',b,36,0)
                different=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
                detail=f' byte_differences={len(different)} first_byte={different[0] if different else -1} sizes={len(a)}/{len(b)}'
        raise RuntimeError('TNVB handoff input/settings mismatch; cached trial was not reused'+detail)
    if (manifest['files']!=inventory(root/'mps') or
            manifest['context']!=inventory_file(root/'context.npz') or
            manifest['response']!=hashlib.sha256(Path(response).read_bytes()).hexdigest()):
        raise RuntimeError('TNVB handoff files changed since ENERGY')
    tags=manifest['state_tags']
    if not tags or len(tags)!=len(set(tags)) or any('/' in t or '\\' in t for t in tags):
        raise RuntimeError('TNVB handoff has invalid state tags')
    os.replace(root/'ready.json',root/'consuming.json')
    return manifest


def read_energy_response(path,nstr,norb):
    with open(path,'rb') as stream:
        if stream.read(8)!=b'CPTNRES3':
            raise ValueError('invalid retained TNVB response')
        header=struct.unpack('<8i',stream.read(32))
        if (header[:5]!=(3,0,nstr,norb,0) or header[5] not in (1,nstr) or
                header[6] not in (0,1) or header[7]):
            raise ValueError('invalid retained TNVB response dimensions')
        roots=header[5]
        def array(count,shape):
            data=np.frombuffer(stream.read(count*8),dtype='<f8').copy()
            if data.size!=count or not np.all(np.isfinite(data)):
                raise ValueError('invalid retained TNVB response data')
            return data.reshape(shape,order='F')
        h=array(nstr*nstr,(nstr,nstr));s=array(nstr*nstr,(nstr,nstr))
        energies=array(roots,(roots,));coefficients=array(nstr*roots,(nstr,roots))
        struct.unpack('<4d',stream.read(32))
        diagnostics=None
        if header[6]:
            values=struct.unpack('<3i2d',stream.read(28))
            diagnostics=dict(zip(('iterations','block_actions','peak_subspace_dimension',
                'maximum_relative_residual','maximum_stationarity_residual'),values))
        if stream.read(1):
            raise ValueError('trailing retained TNVB response data')
    return h,s,energies,coefficients,diagnostics


def resume(directory, request_path, request, sum_bond):
    import pair_core as core
    root=Path(directory)
    manifest=claim(root,request_path,root.parent/'response.bin')
    nstr,norb,nelec,_,bond=request[:5]
    if len(manifest['state_tags'])!=nstr:
        raise ValueError('TNVB handoff structure count mismatch')
    h,s,energies,c,diagnostics=read_energy_response(root.parent/'response.bin',nstr,norb)
    with np.load(root/'context.npz',allow_pickle=False) as context:
        rinv,h1,g2=context['rinv'],context['h1'],context['g2']
    if (rinv.shape!=(norb,norb) or h1.shape!=(norb,norb) or
            g2.shape!=(norb,)*4 or not all(np.all(np.isfinite(x)) for x in (rinv,h1,g2))):
        raise ValueError('invalid TNVB handoff integral context')
    core.validate_structure_metric(h,s)
    core.log_summary(f'[TNVB] repr=MPS/SU2 active={nelec}e/{norb}o nstr={nstr} '
        f'D={bond}/{sum_bond} check=GRADIENT state=HANDOFF d3g=FULL')
    result=core.compute_total_state_rdms(str(root/'mps'),str(root.parent),
        manifest['state_tags'],norb,nelec,bond,sum_bond,0,h,s,rinv,h1,g2,0,0,0,
        selected_eigenpair=(float(energies[0]),c[:,0]),consume_source=True,
        orthogonal_output=True)
    fit_error,energy,coefficient,p,pi,d3g,checks=result
    if abs(energy-energies[0])>1e-10:
        raise RuntimeError('TNVB handoff changed its CI eigenvalue')
    c[:,0]=coefficient
    print(f'[TNVB-HANDOFF] reused structures={nstr} mps_build=skipped '
          'hs=skipped ci=skipped state_copy=0',flush=True)
    return h,s,energies,c,(p,pi,d3g),(
        fit_error,checks['identity_error'],0.0,checks['total_seconds']),diagnostics
