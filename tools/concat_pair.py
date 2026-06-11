import sys, glob, numpy as np
files = sorted(glob.glob(sys.argv[1])); out = sys.argv[2]
keys = ["seq1","seq2","len1","len2","cov1","cov2","y_sF","y_sL"]
acc = {k: [] for k in keys}; names = None
for f in files:
    d = np.load(f, allow_pickle=True); names = d["channel_names"]
    for k in keys: acc[k].append(d[k])
data = {k: np.concatenate(acc[k], 0) for k in keys}
np.savez_compressed(out, channel_names=names, **data)
print(f"concat {len(files)}개 → {out}: {data['seq1'].shape} x2, label {data['y_sF'].shape}")
