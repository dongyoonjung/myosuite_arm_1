# myoArm RL 학습 이미지 — CPU 전용, headless(렌더링 없음).
# GCP spot VM에서 이 이미지를 pull해 M1–M4 학습/데이터생성 실행.
FROM python:3.11-slim

# MuJoCo가 헤드리스로 물리 스텝만 돌 땐 GL 불필요. 최소 시스템 의존만.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git tini && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# torch CPU 먼저(별도 인덱스), 그다음 나머지 고정 버전
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

ENTRYPOINT ["tini", "--"]
CMD ["bash"]
