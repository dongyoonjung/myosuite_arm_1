"""시퀀스 데이터 샤드 병합. 사용: python tools/concat_seq.py 'data/seq/shard_*.npz' data/seq/train.npz"""
import glob
import sys

import numpy as np

files = sorted(glob.glob(sys.argv[1]))
assert files, f"샤드 없음: {sys.argv[1]}"
out = sys.argv[2]
data = [np.load(f, allow_pickle=True) for f in files]
merged = {k: np.concatenate([d[k] for d in data]) for k in ("seq", "length", "cov", "y_sF", "y_sL")}
merged["channel_names"] = data[0]["channel_names"]
merged["chan_blocks"] = data[0]["chan_blocks"]
np.savez_compressed(out, **merged)
print(f"병합 {len(files)}샤드 → {out}: seq{merged['seq'].shape} "
      f"약화비율 {(merged['y_sF'].min(1) < 0.999).mean():.2f}")
