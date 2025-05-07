# local/extract_xvectors_from_data.py
import argparse, os, torchaudio, torch, kaldi_io
from tqdm import tqdm
from models.resnet2 import ResNet101  # VBx 레시피 동일 모듈

def load_wav(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)

def mfcc_cmvn(wav):
    mfcc = torchaudio.compliance.kaldi.mfcc(
        wav.unsqueeze(0), sample_frequency=16000, num_mel_bins=23,
        high_freq=0, low_freq=20, dither=0.0, frame_shift=10, frame_length=25
    )[0].T  # shape (frames, 23)
    cmvn = (mfcc - mfcc.mean(0)) / (mfcc.std(0) + 1e-10)
    return cmvn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir',  required=True)
    args = parser.parse_args()

    model = ResNet101(feat_dim=64, embed_dim=256, squeeze_excitation=False)
    ckpt  = torch.load(args.network, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    ark_path = os.path.join(args.out_dir, 'xvector.ark')
    scp_path = os.path.join(args.out_dir, 'xvector.scp')
    with kaldi_io.open_or_fd(ark_path, 'wb') as ark_fh, \
        open(scp_path, 'w') as scp_fh:

        for line in tqdm(open(os.path.join(args.data_dir, 'wav.scp'))):
            utt, *rest = line.strip().split()
            # Kaldi 파이프 형식: cat path |   또는  <wav-path>
            if rest[-1] == '|':            # cat … | 형식
                wav_path = rest[-2]        # 'path' 부분
            else:
                wav_path = rest[0]         # 이미 경로만 있을 때

            feat = mfcc_cmvn(load_wav(wav_path)).unsqueeze(0).transpose(1, 2)
            with torch.no_grad():
                emb, _, _ = model(feat)                # 1×256

            # --- kaldi_io는 write_vec_flt만 제공 ---
            pos = kaldi_io.write_vec_flt(ark_fh, emb.squeeze(0).numpy(), utt)
            scp_fh.write(f"{utt} {ark_path}:{pos}\n")
    print('xvector.ark/scp 저장 완료')

if __name__ == '__main__': main()
