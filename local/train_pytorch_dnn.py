#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import logging
import os
import re
import shutil
import sys
import traceback
import random
import time
import math

import torch
import tempfile, subprocess
import numpy as np
import kaldiio
from torch.utils.data import Dataset

import torch.nn as nn
# import horovod.torch as hvd
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    import horovod.torch as hvd          # Horovod가 있으면 그대로 사용
    HOROVOD = True
except (ImportError, OSError):           # ImportError·CUDA 심볼 오류 모두 잡기
    HOROVOD = False

    class _DummyHvd:
        # 필수 API만 구현 — 모두 단일-GPU용 no-op
        def init(self): pass
        def shutdown(self): pass
        def rank(self): return 0
        def size(self): return 1
        def local_rank(self): return 0
        class Compression:
            none = None
            fp16 = None
        def DistributedOptimizer(self, opt, **kw):   # 그냥 원본 Optimizer 반환
            return opt
        def broadcast_parameters(self, *a, **k): pass
        def broadcast_optimizer_state(self, *a, **k): pass
    hvd = _DummyHvd()

from utils.pytorch_data import KaldiArkDataset
from utils import ze_utils
from models.metrics import AddMarginProduct, ArcMarginProduct, SphereProduct
from models.resnet2 import *


APEX_AVAILABLE = False

torch.backends.cudnn.benchmark = True

logger = logging.getLogger('train-pytorch')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(module)s:%(lineno)s - %(levelname).1s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    """ Get args from stdin.
    """

    parser = argparse.ArgumentParser(
        description="""Trains a CNN PyTorch model using segment-level 
        objectives like cross-entropy and mean-squared-error (there is only 
        one output for each input segment using statestics pooling layer).
        DNNs include TDNNs and CNNs and should be defined in the 
        models_pytorch.py file as a separate class.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    parser.add_argument("--use-gpu", type=str, dest='use_gpu', choices=["yes", "no"],
                        help="Use GPU for training.", default="yes")

    parser.add_argument("--fp16-compression", type=str, dest='fp16_compression', choices=["yes", "no"],
                        help="Use 16 bit flouting point compression.", default="no")

    parser.add_argument("--apply-cmn", type=str, dest='apply_cmn', choices=["yes", "no"],
                        help="Apply one more CMN on training examples.", default="no")

    parser.add_argument("--momentum", type=float, dest='momentum', default=0.0,
                        help="""Momentum used in update computation.""")

    parser.add_argument("--model", type=str, dest='model', required=True,
                        help="Shows the class name which should be defined in models/*.py file.")

    parser.add_argument("--metric", type=str, dest='metric', default="linear",
                        choices=['linear', 'add_margin', 'arc_margin', 'sphere'],
                        help="Shows the custum metric that should be used as last layer.")

    parser.add_argument("--dir", type=str, required=True,
                        help="The main directory for this experiment to store "
                             "the models and all other files such as logs. The"
                             "trained models will saved in models sub-directory.")
    parser.add_argument("--model-init", type=str, default=None,
                        help="Provided pretrained model")

    parser.add_argument("--egs-dir", type=str, dest='egs_dir', required=True,
                        help="Directory of training egs in Kaldi like. Note that we slightly"
                             "changed the Kaldi egs directory and so you should use our"
                             "script to create the egs_dir.")

    parser.add_argument("--num-epochs", type=int, dest='num_epochs', default=3,
                        help="Number of epochs to train the model.")

    parser.add_argument("--num-targets", type=int, dest='num_targets', required=True,
                        help="Shows the number of output of the neural network, here "
                             "number of speakers in the training data.")
    parser.add_argument("--embed-dim", type=int, dest="embed_dim", default=128,
                        help="The dimension of the speaker embeddings")

    parser.add_argument("--initial-effective-lrate", type=float,
                        dest='initial_effective_lrate', default=0.001,
                        help="Learning rate used during the initial iteration.")

    parser.add_argument("--final-effective-lrate", type=float,
                        dest='final_effective_lrate', default=None,
                        help="Learning rate used during the final iteration.")
    parser.add_argument("--initial-margin-m", type=float, default=None,
                        help="hyper parameter margin used for the initial iteration")

    parser.add_argument("--final-margin-m", type=float, default=None,
                        help="hyper parameter margin used for the final iteration")

    parser.add_argument("--optimizer", type=str, dest='optimizer', default='SGD', choices=['SGD', 'Adam'],
                        help="Optimizer for training.")

    parser.add_argument('--warmup-epochs', type=int, dest='warmup_epochs', default=0,
                        help='number of warmup epochs')

    parser.add_argument("--optimizer-weight-decay", type=float, dest='optimizer_weight_decay',
                        default=0, help="Optimizer weight decay for training.")

    parser.add_argument("--minibatch-size", type=int, dest='minibatch_size', required=True,
                        help="Size of the minibatch used in SGD/Adam training.")

    parser.add_argument("--frame-downsampling", type=int, dest='frame_downsampling', default=0,
                        help="For downsampling frames. This shows number of frames to ignore."
                             "Defult is zero and means no downsampling.")

    parser.add_argument("--random-seed", type=int, dest='random_seed', default=0,
                        help="""Sets the random seed for PyTorch random seed.  Note that we don't
                        shuffle examples for reading speed.  The examples was already shuffles once
                        using Kaldi in preparation stage.  Warning: This random seed does not control
                        all aspects of this experiment.  There might be other random seeds used in 
                        other stages of the experiment like data preparation (e.g. volume perturbation).""")

    parser.add_argument("--preserve-model-interval", dest="preserve_model_interval",
                        type=int, default=100, help="""Determines iterations for which 
                        models will be preserved during cleanup.  If mod(iter, preserve_model_interval) == 0
                        model will be preserved.""")

    parser.add_argument("--cleanup", default=False, action='store_true', help="Clean up models after training.")

    parser.add_argument("--stage", type=int, default=-2,
                        help="Specifies the stage of the training to execution from.")

    parser.add_argument('--use-apex', default=False, action='store_true', help='use APEX if available')

    parser.add_argument('--fix-margin-m', default=None, type=int,
                        help='fix margin m parameter after Nth iteration to its final value')
    
    parser.add_argument('--squeeze-excitation', default=False, action='store_true',
                        help='use squeeze excitation layers')
    
    parser.add_argument("--val-egs-dir", type=str, default=None,
                    help="Directory of validation egs. If not set, "
                         "train_egs_dir/train_diagnostic_egs.*.ark is used.")
    
    parser.add_argument("--trials-path", type=str, default=None,
                    help="eer 출력을 위한 trials txt파일. utils/make_cosine_trials.py 로 생성 가능.")

    args = parser.parse_args()

    args = process_args(args)

    # apex

    try:
        if args.use_apex:
            from apex import amp
        else:
            raise ModuleNotFoundError
        APEX_AVAILABLE = True
    except ModuleNotFoundError:
        APEX_AVAILABLE = False

    return args


def process_args(args):
    """ Process the options got from get_args() """
    models_dir = os.path.join(args.dir, 'models')
    log_dir = os.path.join(args.dir, 'log')
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(os.path.join(args.dir, 'log', 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(' '.join(sys.argv))
    logger.info(f'Running on host: {str(os.uname()[1])}')

    args.use_gpu = args.use_gpu == 'yes'
    args.fp16_compression = args.fp16_compression == 'yes'
    args.apply_cmn = args.apply_cmn == 'yes'

    return args


def train_one_iteration(args, main_dir, _iter, model, data_loader, optimizer, criterion,
                        device, log_interval, len_train_sampler, writer):
    """ Called from train for one iteration of neural network training

    Selected args:
        shrinkage_value: If value is 1.0, no shrinkage is done; otherwise
            parameter values are scaled by this value.
    """
    
    iter_loss = 0
    num_correct = 0
    num_total = 0

    start_time = time.time()
    model.train()
    len_data_loader = len(data_loader)
    total_gpu_waiting = 0
    batch_idx = 0
    try:
        for batch_idx, (data, target) in enumerate(data_loader):
            gpu_waiting = time.time()
            target = target.long()

            if args.frame_downsampling > 0:
                ss = random.randint(0, args.frame_downsampling)
                data = data[:, ss::args.frame_downsampling + 1, :]

            data = data.to(torch.device(device=device))
            target = target.to(torch.device(device=device))
            data = data.transpose(1, 2)
            optimizer.zero_grad()
            output = model(data)
            if args.metric == 'linear':
                output = model.metric(output)
            else:
                output = model.metric(output, target)

            loss = criterion(output, target)

            if APEX_AVAILABLE:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    optimizer.synchronize()
                with optimizer.skip_synchronize():
                    optimizer.step()
            else:
                loss.backward()
                optimizer.step()

            loss = loss.item()
            iter_loss += loss
            predict = output.max(1)[1]
            num_correct += predict.eq(target).sum().item()
            num_total += len(data)
            total_gpu_waiting += time.time() - gpu_waiting
            if writer is not None and hvd.rank() == 0:
                writer.add_scalar('Loss/train', iter_loss, _iter)
                writer.add_scalar('Accuracy/train', num_correct * 1.0 / num_total, _iter)
                global_step = _iter   # 전체 iter 인덱스

                # ① 현재 lr 가져오기 (Scheduler 사용 여부에 따라)
                current_lr = optimizer.param_groups[0]['lr']

                writer.add_scalar('lr', current_lr, global_step)

                # ② weight-decay 항 계산: λ × ‖W‖₂
                wd = optimizer.param_groups[0].get('weight_decay', 0.0)
                if wd > 0:
                    weight_norm = 0.0
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            weight_norm += p.data.norm(2).item()
                    writer.add_scalar('weight_decay|w|', wd * weight_norm, global_step)
            if batch_idx > 0 and batch_idx % log_interval == 0 and hvd.rank() == 0:
                # use train_sampler to determine the number of examples in this worker's partition.
                logger.info(f'Train Iter: {_iter} [{batch_idx * len(data)}/{len_train_sampler} '
                            f'({100.0 * batch_idx / len_data_loader:.0f}%)]  Loss: {loss:.6f}')
    except RuntimeError as e:
        logger.warning(f'RuntimeError when processing mini batch {batch_idx + 1}/{len_data_loader}. '
                       f'If you see this message frequently, it probably means, '
                       f'that there is some problem with your data or code. {os.linesep}{e}')
    except TypeError as e:
        logger.warning(f'TypeError when processing mini batch {batch_idx + 1}/{len_data_loader}. '
                       f'If you see this message frequently, it probably means, '
                       f'that there is some problem with your data or code. {os.linesep}{e}')
    acc = num_correct * 1.0 / num_total
    iter_loss /= len_data_loader

    # save the model and do logging in worker with rank zero
    args.processed_archives += hvd.size()
    if hvd.rank() == 0:
        logger.info(f'Iteration Loss: {iter_loss:.4f}  Accuracy: {acc * 100:.2f}%')
        logger.info(f'Elapsed time: {(time.time() - start_time) / 60.0:.2f} minutes '
                    f'and GPU waiting: {total_gpu_waiting / 60.0:.2f} minutes.')
        new_model_path = os.path.join(main_dir, 'models', f'raw_{_iter}.pth')
        logger.info(f'Saving model to: {new_model_path}')
        saving_model = model
        while isinstance(saving_model, torch.nn.DataParallel):
            saving_model = saving_model.module
        _save_checkpoint(
            {
                'processed_archives': args.processed_archives,
                'class_name': args.model,
                'frame_downsampling': args.frame_downsampling,
                'state_dict': saving_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            file_name=new_model_path)


def eval_network(model, data_loader, device):
    """ Called from train for one iteration of neural network training

    Selected args:
        shrinkage_value: If value is 1.0, no shrinkage is done; otherwise
            parameter values are scaled by this value.
    """

    iter_loss = 0
    num_correct = 0
    num_total = 0

    start_time = time.time()
    model.eval()
    len_data_loader = len(data_loader)
    total_gpu_waiting = 0
    batch_idx = 0
    try:
        for batch_idx, (data, target) in enumerate(data_loader):
            gpu_waiting = time.time()
            target = target.long()
            data = data.to(torch.device(device=device))
            target = target.to(torch.device(device=device))
            data = data.transpose(1, 2)
            output, _, _ = model(data)

            if args.metric == "linear":
                output = model.metric(output)
            else:
                output = model.metric(output, target)
            loss = model.criterion(output, target) + model.get_l2_loss()
            loss = loss.item()
            iter_loss += loss
            predict = output.max(1)[1]
            num_correct += predict.eq(target).sum().item()
            num_total += len(data)
            total_gpu_waiting += time.time() - gpu_waiting
    except RuntimeError:
        logger.warning(f'RuntimeError when processing mini batch {batch_idx + 1}/{len_data_loader}. '
                       f'If you see this message frequently, it probably means, '
                       f'that there is some problem with your data or code.')
    acc = num_correct * 1.0 / num_total
    iter_loss /= len_data_loader

    # save the model and do loging in worker with rank zero
    logger.info('Final Iteration Loss: {:.6f}\t and Accuracy: {:.2f}%'.format(iter_loss, acc * 100))
    logger.info("Elapsed time for processing whole training minibatches is %.2f minutes." %
                ((time.time() - start_time) / 60.0))
    logger.info("GPU waiting for processing whole training minibatches is %.2f minutes." %
                (total_gpu_waiting / 60.0))
    
def eval_once(model, loader, device, criterion):
    model.eval()
    tot_loss, tot_corr, tot = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.long().to(device)
            x = x.transpose(1, 2)
            # out = model.metric(model(x), y) if args.metric != 'linear' else model.metric(model(x))
            
            # pred = model(x.to(device))
            # ── forward ───────────────────────────────────────────
            logits = model(x)                 # 원-로짓
            out    = (model.metric(logits, y)    # margin 층 포함
                      if args.metric != 'linear'
                      else model.metric(logits))
            # ──────────────────────────────────────────────────────

            # ── 디버그 출력 : 첫 배치만 ───────────────────────────
            if batch_idx == 0:
                print("model out dim :", logits.shape[1])
                print("label max/min :", y.min().item(), "/", y.max().item())
                print("top-1 preds   :", logits.argmax(1)[:20].cpu().tolist())
                print("labels sample :", y[:20].cpu().tolist())
            # ──────────────────────────────────────────────────────
            
            loss = criterion(out, y).item()
            tot_loss += loss
            tot_corr += out.argmax(1).eq(y).sum().item()
            tot += y.numel()
    return tot_loss / len(loader), tot_corr / tot

# ─────────────────────────────────────────────────────────
# eval_cosine.py  (train_pytorch_dnn.py 내부 혹은 utils)
# ─────────────────────────────────────────────────────────

class FeatsScpDataset(Dataset):
    def __init__(self, feats_scp, cmn=True):
        self.entries = [line.strip().split()           # [utt, ark:pos]
                        for line in open(feats_scp)]
        self.cmn = cmn

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        utt, ark_pos = self.entries[idx]
        mat = kaldiio.load_mat(ark_pos)               # (T, F)
        if self.cmn:
            mat -= np.mean(mat, axis=0, keepdims=True)
        return torch.tensor(mat, dtype=torch.float32), utt
    
CHUNK = 400                 # 한 세그먼트 길이(프레임)

def collate_fixed(batch):
    """
    batch : List[(feat(T,F) Tensor,  utt_id str)]
    반환  : (B, CHUNK, F) Tensor,  List[str]
    • T ≥ CHUNK → 앞쪽 400프레임 사용
    • T < CHUNK → 반복 padding 으로 400프레임 채움
    """
    feats, utts = zip(*batch)
    fixed = []
    for m in feats:                      # m : Tensor (T, F)
        t = m.size(0)
        if t >= CHUNK:
            fixed.append(m[:CHUNK].clone())          # Tensor 깊은 복사
        else:
            reps = (CHUNK + t - 1) // t              # ceil(CHUNK / t)
            pad  = m.repeat(reps, 1)[:CHUNK].clone() # 반복 뒤 슬라이스
            fixed.append(pad)
    return torch.stack(fixed), list(utts)
    
def _cosine_score(e1: np.ndarray, e2: np.ndarray) -> float:
    """코사인 유사도 스코어"""
    return float(np.dot(e1, e2) /
                 (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))

def _write_scores_labels(scores, labels):
    """
    Kaldi compute-eer / compute_min_dcf.py 가 요구하는 형식:
        <score> <target|nontarget>
    """
    tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    for s, l in zip(scores, labels):
        tag = "target" if l == 1 else "nontarget"
        tmp.write(f"{s:.6f} {tag}\n")
    tmp.flush()
    return tmp

def _kaldi_compute_eer(score_file):
    """compute-eer 바이너리 호출 → EER(float 0~1) 반환"""
    out = subprocess.check_output(["compute-eer", score_file.name], text=True)
    # "Equal error rate is  6.27%" 형식
    eer = float(out.strip().split()[-1].rstrip('%')) / 100.0
    return eer

def _write_score_and_label_files(scores, labels):
    tmp_scores = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    tmp_labels = tempfile.NamedTemporaryFile(mode='w+', delete=False)

    for s, l in zip(scores, labels):
        tmp_scores.write(f'{s:.6f}\n')
        tmp_labels.write('target\n' if l == 1 else 'nontarget\n')

    tmp_scores.flush(); tmp_labels.flush()
    return tmp_scores, tmp_labels

def _write_scores_and_trials(trials, scores):
    # trials  = [(utt1, utt2, label)], label∈{0,1}
    # scores  = [float, ...]  (matched 길이)
    tmp_score = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    tmp_trial = tempfile.NamedTemporaryFile(mode='w+', delete=False)

    for (u1, u2, lab), s in zip(trials, scores):
        tag = 'target' if lab == 1 else 'nontarget'
        tmp_score.write(f'{u1} {u2} {s:.6f}\n')
        tmp_trial.write(f'{u1} {u2} {tag}\n')

    tmp_score.flush(); tmp_trial.flush()
    return tmp_score, tmp_trial

def _kaldi_compute_min_dcf(trials, scores, p_target=0.01):
    scr_f, tri_f = _write_scores_and_trials(trials, scores)
    DCF_SCRIPT = os.path.join(
        os.environ['KALDI_ROOT'],
        'egs/sre08/v1/sid/compute_min_dcf.py'
    )
    try:
        out = subprocess.check_output(
            [DCF_SCRIPT, '--p-target', str(p_target),
             scr_f.name, tri_f.name],
            text=True
        )
        dcf = float(out.strip().split()[-1])
    finally:
        os.unlink(scr_f.name); os.unlink(tri_f.name)
    return dcf

def _norm(uid: str) -> str:
    # 뒤쪽 "-123-400-65" 와 같이 "-숫자-숫자-숫자" 패턴 제거
    return re.sub(r'-\d+-\d+-\d+$', '', uid)

# ─────────────────────────────────────────────────────────
def eval_cosine(model, loader, device,
                trials, writer, step,
                hvd_rank=0, p_target=0.01):
    """
    * 모델을 eval 모드로 두고 valid loader 전체 임베딩 추출
    * trials( [(utt1, utt2, label)] )에 따라 코사인 스코어 계산
    * Kaldi compute-eer / compute_min_dcf.py 로 EER·minDCF 산출
    * TensorBoard 기록 후 EER 반환
    ----------------------------------------------------------------------
    Args:
        model      : nn.Module (x-vector encoder)
        loader     : DataLoader(valid set)
        device     : torch.device
        trials     : List[(utt_id1, utt_id2, int{0,1})]
        writer     : torch.utils.tensorboard.SummaryWriter
        step       : global iteration/epoch count
        hvd_rank   : Horovod/torch.distributed rank (기본 0)
        p_target   : minDCF 계산용 P_target (디폴트 0.01)
    """
    model.eval()
    emb_cache = {}                       # utt_id -> nd.array(embed)
    with torch.no_grad():
        for feats, utt_ids in loader:    # feats: (B, T, F)
            x = feats.to(device).transpose(1, 2)
            vecs = model(x).cpu().numpy()
            for uid_raw, vec in zip(utt_ids, vecs):
                emb_cache[_norm(uid_raw)] = vec

    # rank 0만 평가 지표 계산 (멀티 GPU일 때)
    if hvd_rank != 0:
        return None

    # ── (추가) emb_cache에 실제로 존재하는 utt만 필터 ──────────
    valid_utts = set(emb_cache)          # 이번 검증에서 forward된 ID

    filtered_pairs = [
        (_cosine_score(emb_cache[_norm(u1)], emb_cache[_norm(u2)]), lab)
        for u1, u2, lab in trials
        if _norm(u1) in valid_utts and _norm(u2) in valid_utts
    ]

    missing = len(trials) - len(filtered_pairs)
    if missing and hvd_rank == 0:
        logger.warning(f"[Cosine] {missing} pairs skipped "
                       f"(utt not seen in this eval)")
        
    logger.info(f"[Cosine] emb_cache={len(emb_cache)}  "
            f"trial_ids={len(trials)}  "
            f"matched={len(filtered_pairs)}")

    if not filtered_pairs:               # 모든 쌍이 빠질 일은 드물지만 대비
        return None

    scores, labels = zip(*filtered_pairs)
    filtered_trials = [(u1, u2, lab)             # label은 0/1
                   for (u1, u2, lab), _ in zip(trials, filtered_pairs)]

    # Kaldi 유틸 호출
    tmpfile = _write_scores_labels(scores, labels)
    try:
        eer  = _kaldi_compute_eer(tmpfile)
        dcf = _kaldi_compute_min_dcf(filtered_trials, scores, p_target)
    finally:
        os.unlink(tmpfile.name)          # 임시 파일 삭제

    # TensorBoard 기록
    writer.add_scalar("EER/cos",    eer, step)
    writer.add_scalar("minDCF/cos", dcf, step)

    logger.info(f"[Val] step={step:,}  EER={eer*100:.2f}%  minDCF={dcf:.3f}")
    return eer

def _remove_model(nnet_dir, _iter, preserve_model_interval=100):
    if _iter < 1 or _iter % preserve_model_interval == 0:
        return
    model_path = os.path.join(nnet_dir, 'models', f'raw_{_iter}.pth')
    if os.path.exists(model_path):
        os.remove(model_path)


def _save_checkpoint(state, file_name, is_best=False):
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, 'model_best.pth')


def train(args):
    """ The main function for training.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()
    """
    
    writer = SummaryWriter(log_dir=os.path.join(args.dir, 'tensorboard_logs'))  # TensorBoard writer 초기화

    egs_dir = args.egs_dir
    # initialize horovod
    hvd.init()

    # verify egs dir and extract parameters
    [num_archives, egs_feat_dim, arks_num_egs] = ze_utils.verify_egs_dir(egs_dir)
    assert arks_num_egs > 0

    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(hvd.local_rank())

    if hvd.rank() == 0:
        logger.info(f'Device is: {device}')

    num_jobs = hvd.size()
    if hvd.rank() == 0:
        logger.info(f'Using {num_jobs} training jobs.')
    if num_jobs > num_archives:
        raise ValueError('num_jobs cannot exceed the number of archives in the egs directory')

    init_model_path = args.model_init

    # add metric
    try:
        if args.metric == 'add_margin':
            metric_fc = AddMarginProduct(args.embed_dim, args.num_targets, s=32, m=0.2)
        elif args.metric == 'arc_margin':
            metric_fc = ArcMarginProduct(args.embed_dim, args.num_targets, s=32, m=0.2)
        elif args.metric == 'sphere':
            metric_fc = SphereProduct(args.embed_dim, args.num_targets, m=4)
        else:
            metric_fc = nn.Sequential(nn.BatchNorm1d(args.embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.embed_dim, args.num_targets))

        model = eval(args.model)(feat_dim=egs_feat_dim, embed_dim=args.embed_dim, squeeze_excitation=args.squeeze_excitation)

        # load the init model if exist. This is useful when loading from a pre-trained model
        if init_model_path is not None:
            if hvd.rank() == 0:
                logger.info(f'Loading the initial network from: {init_model_path}')
            checkpoint = torch.load(init_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    except AttributeError:
        raise AttributeError(f'The specified class name {args.model} does not exist.')

    model.add_module('metric', metric_fc)

    # move model to device before creating optimizer
    model = model.to(torch.device(device=device))

    parameters = model.parameters()
    
    # ── 1) 백본 동결 ────────────────────────────────
    freeze_prefixes = ('conv1', 'bn1', 'layer1', 'layer2', 'layer3')

    for name, p in model.named_parameters():
        # metric·embedding·layer4 만 학습
        if name.startswith(freeze_prefixes):
            p.requires_grad = False
        else:
            p.requires_grad = True
            
    # ── 2) Optimizer에 학습 가능한 파라미터만 전달 ──
    train_params = [p for p in model.parameters() if p.requires_grad]
    
    print("=== Trainable layers ===")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)

    # # scale learning rate by the number of GPUs.
    # if args.optimizer == 'SGD':
    #     main_optimizer = torch.optim.SGD(parameters, lr=args.initial_effective_lrate * num_jobs,
    #                                      momentum=args.momentum, weight_decay=args.optimizer_weight_decay,
    #                                      nesterov=True)
    #     scheduler = CosineAnnealingWarmRestarts(main_optimizer,
    #                                     T_0=int(num_archives/num_jobs), T_mult=2)
    # elif args.optimizer == 'Adam':
    #     main_optimizer = torch.optim.Adam(parameters, lr=args.initial_effective_lrate * num_jobs,
    #                                       weight_decay=args.optimizer_weight_decay)
    # else:
    #     raise ValueError(f'Invalid optimizer {args.optimizer}.')
    
    if args.optimizer == 'SGD':
        main_optimizer = torch.optim.SGD(train_params, lr=args.initial_effective_lrate * num_jobs,
                                         momentum=args.momentum, weight_decay=args.optimizer_weight_decay,
                                         nesterov=True)
        scheduler = CosineAnnealingWarmRestarts(main_optimizer,
                                        T_0=int(num_archives/num_jobs), T_mult=2)
    elif args.optimizer == 'Adam':
        main_optimizer = torch.optim.Adam(train_params, lr=args.initial_effective_lrate * num_jobs,
                                          weight_decay=args.optimizer_weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {args.optimizer}.')

    if hvd.rank() == 0:
        logger.info(str(model))
        logger.info(str(main_optimizer))

    processed_archives = 0
    if init_model_path is not None and hvd.rank() == 0:
        logger.info(f'Saving the initial network to: {init_model_path}')
        _save_checkpoint({
            'processed_archives': processed_archives,
            'class_name': args.model,
            'frame_downsampling': args.frame_downsampling,
            'state_dict': model.state_dict(),
            'optimizer': main_optimizer.state_dict(),
        }, file_name=init_model_path)

    # save kaldi's files
    if hvd.rank() == 0:
        with open(os.path.join(args.dir, 'models', 'model_info'), 'wt') as fid:
            fid.write(f'{args.model} {egs_feat_dim} {args.num_targets}{os.linesep}')
        with open(os.path.join(args.dir, 'models', 'config.txt'), 'wt') as fid:
            fid.write(str(model))
        with open(os.path.join(args.dir, 'command.sh'), 'wt') as fid:
            fid.write(' '.join(sys.argv) + '\n')
        with open(os.path.join(args.dir, 'max_chunk_size'), 'wt') as fid:
            fid.write(f'10000{os.linesep}')
        with open(os.path.join(args.dir, 'min_chunk_size'), 'wt') as fid:
            fid.write(f'25{os.linesep}')

    # find the last saved model and load it
    saved_models = glob.glob(os.path.join(args.dir, 'models', 'raw_*.pth'))
    finished_iterations = 0
    try:
        for name in saved_models:
            model_id = int(re.split('[_\.]', name)[-2])
            if model_id > finished_iterations:
                finished_iterations = model_id
    except Exception:
        pass

    if finished_iterations > 0:
        model_path = os.path.join(args.dir, 'models', f'raw_{finished_iterations}.pth')
        if hvd.rank() == 0:
            logger.info('Loading model from ' + model_path)

        checkpoint = torch.load(model_path, map_location='cpu')
        processed_archives = checkpoint['processed_archives']
        model.load_state_dict(checkpoint['state_dict'])
        main_optimizer.load_state_dict(checkpoint['optimizer'])

    # note: here minibatch is the size before
    train_dataset = KaldiArkDataset(egs_dir=egs_dir, num_archives=num_archives, num_workers=num_jobs, rank=hvd.rank(),
                                    num_examples_in_each_ark=arks_num_egs, finished_iterations=finished_iterations,
                                    processed_archives=processed_archives, apply_cmn=args.apply_cmn, return_utt=False)

    # use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=num_jobs, rank=hvd.rank())

    kwargs = {'drop_last': False, 'pin_memory': False}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.minibatch_size, 
        sampler=train_sampler, 
        **kwargs)
    
    trials = []
    with open(args.trials_path) as f:
        for line in f:
            u1, u2, lab = line.rstrip().split()
            trials.append((u1, u2, int(lab)))
    # val_ark = os.path.join(egs_dir, 'train_diagnostic_egs.1.ark')
    # val_scp = os.path.join(egs_dir, 'train_diagnostic_egs.1.scp')
    val_egs_dir = args.val_egs_dir if args.val_egs_dir else args.egs_dir
    [val_num_archives, val_egs_feat_dim, val_arks_num_egs] = ze_utils.verify_egs_dir(val_egs_dir)
    
    val_dataset = FeatsScpDataset(
        feats_scp="data/all_combined_aug_and_clean_valid/feats.scp",
        cmn=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.minibatch_size,
        shuffle=False,     # 매번 같은 순서 → 재현성
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fixed          # ★ 핵심
    )
    
    best_eer = 1.0
    
    # val_dataset = KaldiArkDataset(
    #     egs_dir=val_egs_dir,
    #     num_archives=val_num_archives,          # ← 파일 개수로 설정
    #     num_workers=1,
    #     rank=0,                       # validation 은 분산학습 rank 무관
    #     num_examples_in_each_ark=val_arks_num_egs,  # 무시돼도 큰 문제 없음
    #     finished_iterations=0,
    #     processed_archives=0,
    #     apply_cmn=args.apply_cmn,
    #     return_utt=True
    # )
    
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.minibatch_size,
    #     shuffle=False
    # )

    # (optional) compression algorithm. TODO add better compression method
    compression = hvd.Compression.fp16 if args.fp16_compression else hvd.Compression.none
    #
    
    # ── trainable 이름·파라미터만 뽑기 ─────────────────────
    named_trainables = [
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    ]
    
    # wrap optimizer with DistributedOptimizer.
    # optimizer = hvd.DistributedOptimizer(main_optimizer, named_parameters=model.named_parameters(),
    #                                      compression=compression)
    
    optimizer = hvd.DistributedOptimizer(main_optimizer, named_parameters=named_trainables,
                                         compression=compression)

    # broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    criterion = nn.CrossEntropyLoss()
    args.processed_archives = processed_archives

    # set num_iters so that as close as possible, we process the data
    num_iters = int(args.num_epochs * num_archives * 1.0 / num_jobs)
    num_archives_to_process = int(num_iters * num_jobs)

    # initialize APEX
    if APEX_AVAILABLE:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale='dynamic')

    if hvd.rank() == 0:
        logger.info(f'Training will run for {args.num_epochs} epochs = {num_iters} iterations')

    for _iter in range(finished_iterations, num_iters):
        percent = args.processed_archives * 100.0 / num_archives_to_process
        epoch = (args.processed_archives * float(args.num_epochs) / num_archives_to_process)

        effective_learning_rate = args.initial_effective_lrate

        if args.final_effective_lrate is not None:
            effective_learning_rate = math.exp(args.processed_archives * math.log(args.final_effective_lrate /
                                                                                  args.initial_effective_lrate)
                                               / num_archives_to_process) \
                                      * args.initial_effective_lrate

        if args.metric != 'linear':
            effective_margin_m = args.initial_margin_m
            if args.final_margin_m is not None or args.fix_margin_m is not None:
                if args.fix_margin_m is not None:
                    if epoch > args.fix_margin_m:
                        # keep margin fixed
                        num_archives_to_process_margin_m = args.processed_archives
                    else:
                        num_archives_to_process_margin_m = num_archives_to_process - (num_archives_to_process / args.num_epochs) * (args.num_epochs - args.fix_margin_m)
                else:
                    num_archives_to_process_margin_m = num_archives_to_process 
                effective_margin_m = math.exp(args.processed_archives * math.log(args.final_margin_m / args.initial_margin_m)
                                              / num_archives_to_process_margin_m) * args.initial_margin_m
            model.metric.m = effective_margin_m

        # coeff = num_jobs
        # if _iter < args.warmup_epochs > 0 and num_jobs > 1:
        #     coeff = float(_iter) * (num_jobs - 1) / args.warmup_epochs + 1.0
        
        iters_per_epoch = int(num_archives / num_jobs)
        warmup_iters = args.warmup_epochs * iters_per_epoch   # 에포크 → iter 환산
        
        coeff = 1.0
        if _iter < warmup_iters and warmup_iters > 0:
            coeff = float(_iter + 1) / warmup_iters          # 0 → 1 선형 증가

        for param_group in optimizer.param_groups:
            param_group['lr'] = effective_learning_rate * coeff

        if hvd.rank() == 0:
            lr = optimizer.param_groups[0]['lr']

            if args.metric == 'linear':
                logger.info(f'Iter: {_iter + 1}/{num_iters}  Epoch: {epoch:0.2f}/{args.num_epochs:0.1f}'
                            f'  ({percent:0.1f}% complete)  lr: {lr:0.5f}')
            else:
                logger.info(f'Iter: {_iter + 1}/{num_iters}  Epoch: {epoch:0.2f}/{args.num_epochs:0.1f}'
                            f'  ({percent:0.1f}% complete)  lr: {lr:0.5f} margin: {model.metric.m:0.4f}')
                
        train_dataset.set_iteration(_iter + 1)

        train_one_iteration(
            args=args,
            main_dir=args.dir,
            _iter=_iter + 1,
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_interval=500,
            len_train_sampler=len(train_sampler),
            writer=writer)
                    
        # if (_iter + 1) % 5 == 0 and hvd.rank() == 0:  # 5 iter마다 검증
        # if hvd.rank() == 0:  # TODO : validation 테스트용 
        #     val_dataset.set_iteration(_iter + 1)            
        #     v_loss, v_acc = eval_once(model, val_loader, device, criterion)
        #     writer.add_scalar('Loss/val', v_loss, _iter + 1)
        #     writer.add_scalar('Accuracy/val',  v_acc,  _iter + 1)
        #     logger.info(f'Validation  Loss: {v_loss:.4f}  Accuracy: {v_acc*100:.2f}%')
        if (_iter + 1) % 5 == 0 and hvd.rank() == 0:  # 5 iter마다 검증
            # val_dataset.set_iteration(_iter + 1)
            eer = eval_cosine(
                model, val_loader, device,
                trials, writer, _iter + 1,
                hvd_rank=hvd.rank() if hvd else 0
            )
            if hvd.rank() == 0 and eer is not None and eer < best_eer:
                best_eer = eer
                torch.save(model.state_dict(), os.path.join(args.dir, 'models', 'model_best.pth'))
                
        if _iter >= warmup_iters:
            scheduler.step()

        if args.cleanup and hvd.rank() == 0:
            # do a clean up everything but the last 2 models, under certain conditions
            _remove_model(args.dir, _iter - 2, args.preserve_model_interval)
            
    
    if hvd.rank() == 0:
        ze_utils.force_symlink(f'raw_{num_iters}.pth', os.path.join(args.dir, 'models', 'model_final'))

    if args.cleanup and hvd.rank() == 0:
        logger.info(f'Cleaning up the experiment directory {args.dir}')
        for _iter in range(num_iters - 2):
            _remove_model(args.dir, _iter, args.preserve_model_interval)
            
    writer.close()  # TensorBoard writer 닫기


if __name__ == '__main__':
    args = get_args()

    assert os.path.isdir(args.egs_dir), f'egs directory `{args.egs_dir}` does not exist.'

    try:
        train(args)
        ze_utils.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        if os.path.exists('using_gpus.txt') and hvd.rank() == 0:
            logger.info('Removing using_gpus.txt file')
            os.remove('using_gpus.txt')
        sys.exit(1)

