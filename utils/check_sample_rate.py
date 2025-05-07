import os
import soundfile as sf
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 오디오 파일 확장자 목록
VALID_EXTENSIONS = (".wav", ".flac", ".mp3")

def is_audio_file(filename):
    return filename.lower().endswith(VALID_EXTENSIONS)

def get_audio_files(root_dir):
    file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if is_audio_file(filename):
                file_list.append(os.path.join(dirpath, filename))
    return file_list

def check_sample_rate(filepath):
    try:
        info = sf.info(filepath)
        if info.samplerate != 16000:
            return filepath, info.samplerate
    except Exception as e:
        return f"[ERROR] {filepath}: {e}", None
    return None

def find_non_16k_files_parallel(file_list, num_workers):
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(check_sample_rate, file_list), total=len(file_list)))
    return [res for res in results if res is not None]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find audio files not sampled at 16kHz.")
    parser.add_argument("--folder", type=str, help="Path to root folder to search")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    print(f"📂 Searching in: {args.folder}")
    file_list = get_audio_files(args.folder)
    print(f"🔍 Found {len(file_list)} audio files to check.")

    results = find_non_16k_files_parallel(file_list, args.workers)

    if results:
        print(f"\n⚠️ {len(results)} file(s) not sampled at 16kHz:")
        for path, sr in results:
            print(f"{path} (sample rate: {sr})")
    else:
        print("✅ All audio files are 16kHz.")
