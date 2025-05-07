#!/usr/bin/env python
"""
all_combined_aug_and_clean_valid/utt2spk → cosine_trials.txt
  --anchor-per-spk  화자당 anchor 개수 (positive용)
  --neg-per-spk     화자당 negative 쌍 수
"""

import random, itertools, argparse, pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--utt2spk", required=True,
                    help="…/utt2spk 경로")
parser.add_argument("--out-dir", default=None,
                    help="출력 폴더(기본: utt2spk 폴더)")
parser.add_argument("--out-name", default="cosine_trials.txt")
parser.add_argument("--anchor-per-spk", type=int, default=5)
parser.add_argument("--neg-per-spk", type=int, default=20)
parser.add_argument("--utt-list", default=None,
                    help="egs 단계에서 실제로 사용된 utt 리스트 파일")
args = parser.parse_args()

utt2spk_path = pathlib.Path(args.utt2spk)
out_dir = pathlib.Path(args.out_dir) if args.out_dir else utt2spk_path.parent
out_path = out_dir / args.out_name
out_dir.mkdir(parents=True, exist_ok=True)

# 1) 허용 utt 집합 먼저 읽기 ──────────────────────────
allowed = None
if args.utt_list:
    allowed = set(pathlib.Path(args.utt_list).read_text().split())

# 2) utt2spk 읽기 (+ allowed 필터) ───────────────────
utt2spk = {}
for line in utt2spk_path.read_text().splitlines():
    utt, spk = line.split()
    if allowed is not None and utt not in allowed:
        continue                      # ★ egs에 없는 utt 제거
    utt2spk.setdefault(spk, []).append(utt)

trials = []

# 2) positive
for spk, utts in utt2spk.items():
    random.shuffle(utts)
    anchors = utts[: min(len(utts), args.anchor_per_spk)]
    for u1, u2 in itertools.combinations(anchors, 2):
        trials.append((u1, u2, 1))

# 3) negative
spk_list = list(utt2spk)
for spk in spk_list:
    for _ in range(args.neg_per_spk):
        other = random.choice([s for s in spk_list if s != spk])
        trials.append((
            random.choice(utt2spk[spk]),
            random.choice(utt2spk[other]),
            0
        ))

random.shuffle(trials)
with out_path.open("w") as f:
    for u1, u2, lab in trials:
        f.write(f"{u1} {u2} {lab}\n")

print(f"[OK] {len(trials):,} trials → {out_path}")
