#!/usr/bin/env bash
set -e

export KALDI_ROOT=/dataset/kaldi
export PATH=$KALDI_ROOT/src/ivectorbin:$KALDI_ROOT/src/bin:$PATH

: "${VBX_ROOT:?Need to export VBX_ROOT}"
. $VBX_ROOT/utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "사용: $0 <network.pth> <data-dir> <xvector-dir>" && exit 1
net=$1; data=$2; dir=$3
mkdir -p "$dir/log"

python3 $VBX_ROOT/local/extract_xvectors_from_data.py \
        --network "$net" --data-dir "$data" --out-dir "$dir"  || exit 1
echo "xvector 추출 완료 → $dir/xvector.scp"
