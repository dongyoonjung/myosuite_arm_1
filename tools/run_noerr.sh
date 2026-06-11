#!/bin/bash
set -e
cd /home/aaron/projects/myoarm_1
V=.venv/bin/python
echo "### $(date) [1/2] KIN 데이터 생성(M3v·rise·fPCA·노이즈, mix, 12샤드)"
mkdir -p data/seqn; rm -f data/seqn/shard_*.npz
for i in $(seq 0 11); do
  OMP_NUM_THREADS=1 $V gen_seq_data.py --model models/ppo_arm_M3v.zip \
    --vecnorm models/ppo_arm_M3v_vec.pkl --task mix --n 1000 --seed $i \
    --ref-gen fpca --kin-only 1 --motor-noise 0.10 --kin-noise 0.0087 \
    --out data/seqn/shard_$i.npz > /tmp/ne_$i.log 2>&1 &
done
wait
$V tools/concat_seq.py 'data/seqn/shard_*.npz' data/seqn/train.npz
rm -f data/seqn/shard_*.npz
echo "### $(date) [2/2] KIN 회귀 (err 제외 = 관절각+속도만, 철학 정합)"
OMP_NUM_THREADS=20 $V regress_seq.py --data data/seqn/train.npz --channels all --no-err \
  --epochs 120 --threads 20 --out models/regressor_kin_noerr.pt > /tmp/G_noerr.log 2>&1
echo "### $(date) NOERR DONE" | tee /tmp/noerr.done
