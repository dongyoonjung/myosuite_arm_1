#!/bin/bash
set -e
cd /home/aaron/projects/myoarm_1
V=.venv/bin/python
echo "### $(date) [1/2] 페어 데이터 생성(8000 피험자 = 16k 롤아웃, 12샤드)"
mkdir -p data/pair; rm -f data/pair/shard_*.npz
for i in $(seq 0 11); do
  OMP_NUM_THREADS=1 $V gen_pair.py --model models/ppo_arm_M3v.zip \
    --vecnorm models/ppo_arm_M3v_vec.pkl --n 667 --seed $i \
    --ref-gen fpca --motor-noise 0.10 --kin-noise 0.0087 \
    --out data/pair/shard_$i.npz > /tmp/pr_$i.log 2>&1 &
done
wait
$V tools/concat_pair.py 'data/pair/shard_*.npz' data/pair/train.npz
rm -f data/pair/shard_*.npz
echo "### $(date) [2/2] 페어 회귀(T1+T2 공유 θ, 어텐션 풀링, err 제외)"
OMP_NUM_THREADS=20 $V regress_pair.py --data data/pair/train.npz \
  --epochs 120 --threads 20 --out models/regressor_pair.pt > /tmp/G_pair.log 2>&1
grep -E "KIN.*채널|평균 R²" /tmp/G_pair.log
echo "### $(date) PAIR DONE" | tee /tmp/pair.done
