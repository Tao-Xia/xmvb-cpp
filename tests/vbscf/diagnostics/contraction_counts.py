"""Exact prefix-DP census. No amplitudes, product arrays or numerical cutoff."""
from collections import defaultdict
import json
from math import comb

def census(n, diagonal=False):
    r = n // 2
    states = {(0, 0, -1, -1): 1}
    for k in range(n):
        out = defaultdict(int)
        for (a, b, first, last), count in states.items():
            for da, db in ((0, 0), (1, 1)) if diagonal else ((0, 0), (0, 1), (1, 0), (1, 1)):
                aa, bb = a + da, b + db
                if aa > r or bb > r or aa + n - k - 1 < r or bb + n - k - 1 < r:
                    continue
                height = aa + bb - (k + 1)
                if height < -1:
                    continue
                f, l = first, last
                if height < 0:
                    f = k if f < 0 else f
                    l = k
                out[aa, bb, f, l] += count
        states = out
    return {(f, l): v for (a, b, f, l), v in states.items() if a == b == r}

rows = []
for n in (12, 14, 18):
    both = census(n)
    diagonal = census(n, True)
    u = comb(n, n // 2)
    p = n * (n + 1) // 2
    for parity in (1, -1):
        hist={key:(value+parity*diagonal.get(key,0))//2 for key,value in both.items()}
        flops=2*sum(v*(p if f<0 else (f+1)*(n-1-l))**2 for (f,l),v in hist.items())
        full=2*sum(v*(p if f<0 else (f+1)*(n-1-l))*p for (f,l),v in hist.items())
        rank = u // 2
        factor_mix = 2 * rank * (2 * u) * p * p
        projection = 2 * rank * u * (p + 3)
        rows.append(dict(n=n,parity=parity,strings=u,source_support=both.get((-1,-1),0),
                         targets=sum(hist.values()),source_factors=rank,
                         product_mix_flops=flops,product_full_channel_flops=full,
                         factor_mix_flops=factor_mix,factor_projection_dot_flops=projection,
                         factor_mix_ratio=factor_mix/flops))
print(json.dumps(rows, indent=2))
