import os
import shutil
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

def process_folder(folder_path, input_root, output_root, max_files=20):
    """한 폴더를 처리: 랜덤하게 20개 선택하고 복사"""
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    if not wav_files:
        return

    if len(wav_files) <= max_files:
        selected_files = wav_files
    else:
        selected_files = random.sample(wav_files, max_files)

    # 출력 폴더 만들기 (input_root 기준 상대경로로 계산)
    relative_path = os.path.relpath(folder_path, input_root)
    output_folder = os.path.join(output_root, relative_path)
    os.makedirs(output_folder, exist_ok=True)

    # 파일 복사
    for filename in selected_files:
        src = os.path.join(folder_path, filename)
        dst = os.path.join(output_folder, filename)
        shutil.copy2(src, dst)

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    max_files = args.max_files
    num_workers = args.num_workers

    # 모든 하위 폴더 재귀적으로 찾기
    all_subfolders = []
    for root, dirs, files in os.walk(input_dir):
        if any(file.endswith('.wav') for file in files):
            all_subfolders.append(root)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(lambda f: process_folder(f, input_dir, output_dir, max_files), all_subfolders),
                  total=len(all_subfolders),
                  desc="Processing folders"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='원본 kaldi style segments 최상위 폴더')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='20개로 제한된 파일들을 저장할 새 경로')
    parser.add_argument('--max_files', type=int, default=20,
                        help='각 폴더당 남길 최대 파일 수')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='병렬 처리할 워커 수 (스레드 수)')
    args = parser.parse_args()

    main(args)
