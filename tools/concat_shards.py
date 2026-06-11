"""M4 데이터 샤드 병합. 사용: python tools/concat_shards.py 'data/sim/shard_*.npz' data/sim/train.npz"""
import glob
import sys

import numpy as np

files = sorted(glob.glob(sys.argv[1]))
assert files, f"샤드 없음: {sys.argv[1]}"
out = sys.argv[2]
data = [np.load(f, allow_pickle=True) for f in files]
merged = {k: np.concatenate([d[k] for d in data]) for k in ("X", "y_sF", "y_sL", "latent")}
merged["feature_names"] = data[0]["feature_names"]
merged["channel_names"] = data[0]["channel_names"]
np.savez_compressed(out, **merged)
print(f"병합 {len(files)}샤드 → {out}: X{merged['X'].shape} "
      f"약화비율 {(merged['y_sF'].min(1) < 0.999).mean():.2f}")
