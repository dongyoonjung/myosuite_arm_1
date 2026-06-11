#!/bin/bash
set -e
cd /home/aaron/projects/myoarm_1
V=.venv/bin/python
echo "### $(date) [1/3] 손목+mix(T1+T2) 학습 fresh 6M"
OMP_NUM_THREADS=20 $V train.py --task mix --latent-mode sample --perturb --wrist --horizon 4.0 \
  --motor-noise 0.10 --curriculum-k 1,2,3 --curriculum-w 0.5,0.3,0.2 \
  --timesteps 6000000 --n-envs 20 --out ppo_arm_wrist_mix --gate-every 2000000 \
  --n-steps 512 --batch-size 4096 > /tmp/wm_train.log 2>&1
echo "### $(date) [2/3] KIN-only 데이터(16샤드, mix)"
mkdir -p data/seqwm; rm -f data/seqwm/shard_*.npz
for i in $(seq 0 15); do
  OMP_NUM_THREADS=1 $V gen_seq_data.py --model models/ppo_arm_wrist_mix.zip \
    --vecnorm models/ppo_arm_wrist_mix_vec.pkl --task mix --n 1000 --seed $i \
    --wrist --horizon 4.0 --kin-only 1 --motor-noise 0.10 --kin-noise 0.0087 \
    --out data/seqwm/shard_$i.npz > /tmp/wmsh_$i.log 2>&1 &
done
wait
$V tools/concat_seq.py 'data/seqwm/shard_*.npz' data/seqwm/train.npz
rm -f data/seqwm/shard_*.npz
echo "### $(date) [3/3] KIN-only 시계열 회귀(손목 12채널)"
OMP_NUM_THREADS=20 $V regress_seq.py --data data/seqwm/train.npz --channels all \
  --epochs 120 --threads 20 --out models/regressor_wrist_mix.pt > /tmp/wmG.log 2>&1
grep "평균 R²" /tmp/wmG.log
echo "### $(date) WRIST-MIX DONE" | tee /tmp/wm.done
