#!/bin/bash
# 더 그럴듯한(현실적) 데이터 생성·평가 파이프라인 (문헌 기반):
#  - 신호의존 운동노이즈(Harris&Wolpert): σ∝명령
#  - EMG 측정노이즈 + 운동학 센서노이즈(가법)
#  - 표면EMG 제한관측(심부 회전근개 제외) vs privileged 전체
# 백그라운드 자율 실행. 결과 로그: /tmp/noisyG_*.log, 요약: /tmp/noisy_pipeline.done
set -e
cd /home/aaron/projects/myoarm_1
V=.venv/bin/python
MN=0.10; EMG=0.03; KIN=0.0087   # 운동노이즈/EMG노이즈/운동학노이즈(rad≈0.5°)

echo "### $(date) noisy seq data gen 시작 (M3b + motor/obs noise)"
mkdir -p data/seq_noisy; rm -f data/seq_noisy/shard_*.npz
for i in $(seq 0 19); do
  OMP_NUM_THREADS=1 $V gen_seq_data.py --model models/ppo_arm_M3b.zip \
    --vecnorm models/ppo_arm_M3b_vec.pkl --task mix --n 1000 --seed $i \
    --motor-noise $MN --emg-noise $EMG --kin-noise $KIN \
    --out data/seq_noisy/shard_$i.npz > /tmp/nsh_$i.log 2>&1 &
done
wait
$V tools/concat_seq.py 'data/seq_noisy/shard_*.npz' data/seq_noisy/train.npz
echo "### $(date) 회귀 학습(노이즈 데이터): all/act/kin/surf+kin"
for SPEC in "all:all" "act:act" "kin:kin" "surfkin:surf+kin"; do
  NAME=${SPEC%%:*}; CH=${SPEC##*:}
  OMP_NUM_THREADS=20 $V regress_seq.py --data data/seq_noisy/train.npz \
    --channels "$CH" --epochs 100 --threads 20 --out models/regressor_noisy_$NAME.pt \
    > /tmp/noisyG_$NAME.log 2>&1
  echo "  [$NAME] $(grep '평균 R²' /tmp/noisyG_$NAME.log | tr '\n' '  ')"
done
echo "### $(date) PIPELINE DONE" | tee /tmp/noisy_pipeline.done
