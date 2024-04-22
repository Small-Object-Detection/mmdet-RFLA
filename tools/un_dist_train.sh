CONFIG=$1
GPU_ID=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --gpu-id=$GPU_ID --launcher pytorch ${@:3}

python tools/analysis_tools/analyze_results.py \
       result_rfla/aitodv2_cascade_r50_rfla_kld_1x.py \
       result_rfla/result.pkl \
       result_new \
       --show-score-thr 0.3