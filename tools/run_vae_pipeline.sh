#!/bin/bash
# 새 방법론 전체 파이프라인:
#  - 참조 = KIMHu 궤적 VAE 생성모델(ref_gen=vae, 평균+1PC 워핑 대체)
#  - 운동노이즈(Harris&Wolpert) 하 정책 재적응
#  - KIN(운동학)만으로 근육 파라미터 식별(EMG/ACT 사용 안 함)
# 백그라운드 자율. 결과: /tmp/vae_pipeline.done, /tmp/kinvae_G.log
set -e
cd /home/aaron/projects/myoarm_1
V=.venv/bin/python
MN=0.10; KIN=0.0087

echo "### $(date) [1/3] M3v 재학습 (M3b warm-start, VAE refs + 운동노이즈 + 섭동)"
OMP_NUM_THREADS=20 $V train.py --task mix --latent-mode sample --perturb \
  --curriculum-k 1,2,3 --curriculum-w 0.5,0.3,0.2 --ref-gen vae --motor-noise $MN \
  --timesteps 4000000 --n-envs 20 --out ppo_arm_M3v --gate-every 1000000 \
  --resume models/ppo_arm_M3b.zip --vecnorm models/ppo_arm_M3b_vec.pkl > /tmp/M3v.log 2>&1
echo "### $(date) M3v 완료: $(grep 'gate @' /tmp/M3v.log | tail -1)"

echo "### $(date) [2/3] KIN-only 데이터 생성 (20샤드, VAE refs + 노이즈)"
mkdir -p data/seqv; rm -f data/seqv/shard_*.npz
for i in $(seq 0 19); do
  OMP_NUM_THREADS=1 $V gen_seq_data.py --model models/ppo_arm_M3v.zip \
    --vecnorm models/ppo_arm_M3v_vec.pkl --task mix --n 1000 --seed $i \
    --ref-gen vae --kin-only 1 --motor-noise $MN --kin-noise $KIN \
    --out data/seqv/shard_$i.npz > /tmp/vsh_$i.log 2>&1 &
done
wait
$V tools/concat_seq.py 'data/seqv/shard_*.npz' data/seqv/train.npz
rm -f data/seqv/shard_*.npz                       # 병합 후 샤드 삭제(디스크)

echo "### $(date) [3/3] KIN-only 시계열 회귀(TCN)"
OMP_NUM_THREADS=20 $V regress_seq.py --data data/seqv/train.npz --channels all \
  --epochs 120 --threads 20 --out models/regressor_kinvae.pt > /tmp/kinvae_G.log 2>&1
echo "  $(grep '평균 R²' /tmp/kinvae_G.log | tr '\n' '  ')"
echo "### $(date) VAE-PIPELINE DONE" | tee /tmp/vae_pipeline.done
