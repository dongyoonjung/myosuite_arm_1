#!/bin/bash
# 손목 학습 완료 대기 → KIN-only 손목 데이터 생성 → 시계열 회귀(손목근 식별 포함).
set -e
cd /home/aaron/projects/myoarm_1
V=.venv/bin/python
until [ -f models/ppo_arm_wrist.zip ]; do sleep 20; done
sleep 5
echo "### $(date) [1/2] KIN-only 손목 데이터 생성(16샤드, full-seq fPCA, 운동노이즈)"
mkdir -p data/seqw; rm -f data/seqw/shard_*.npz
for i in $(seq 0 15); do
  OMP_NUM_THREADS=1 $V gen_seq_data.py --model models/ppo_arm_wrist.zip \
    --vecnorm models/ppo_arm_wrist_vec.pkl --task T1 --n 1000 --seed $i \
    --wrist --horizon 4.0 --kin-only 1 --motor-noise 0.10 --kin-noise 0.0087 \
    --out data/seqw/shard_$i.npz > /tmp/wsh_$i.log 2>&1 &
done
wait
$V tools/concat_seq.py 'data/seqw/shard_*.npz' data/seqw/train.npz
rm -f data/seqw/shard_*.npz
echo "### $(date) [2/2] KIN-only 시계열 회귀(손목 12채널)"
OMP_NUM_THREADS=20 $V regress_seq.py --data data/seqw/train.npz --channels all \
  --epochs 120 --threads 20 --out models/regressor_wrist.pt > /tmp/wristG.log 2>&1
grep "평균 R²" /tmp/wristG.log
echo "### $(date) WRIST PIPELINE DONE" | tee /tmp/wrist_post.done
