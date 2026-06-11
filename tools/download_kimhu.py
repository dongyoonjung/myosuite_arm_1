"""KIMHu 스켈레톤 CSV만 다운로드(EMG·이미지 제외). 매니페스트에서 fileId·경로 파싱.
사용: python tools/download_kimhu.py /home/aaron/kimhu_manifest.json"""
import os
import re
import sys
import time
import urllib.request

manifest = sys.argv[1] if len(sys.argv) > 1 else "/home/aaron/kimhu_manifest.json"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kimhu")
lines = [l.strip() for l in open(manifest) if "skeleton_tracking.csv" in l]
print(f"스켈레톤 CSV {len(lines)}개")
ok = 0
for i, url in enumerate(lines):
    fid = re.search(r"fileId=([a-f0-9]+)", url).group(1)
    pm = re.search(r"path=(/V2/[^&]+)", url)
    rel = pm.group(1).lstrip("/")                      # V2/MAYR02 T1/MAYR02_T1_skeleton_tracking.csv
    dst = os.path.join(ROOT, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 1e6:
        ok += 1; continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    clean = f"https://download.scidb.cn/download?fileId={fid}"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(clean, dst)
            sz = os.path.getsize(dst)
            if sz > 1e6:
                ok += 1
                print(f"  [{i+1}/{len(lines)}] {rel.split('/')[1]} {sz//1024//1024}MB")
                break
        except Exception as e:
            print(f"  [{i+1}] 재시도 {attempt}: {e}"); time.sleep(3)
print(f"완료 {ok}/{len(lines)}")
