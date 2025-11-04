#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assign3_task1.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit


# PREPARE DATA
python preprocess_shared_voc.py \
    --source-lang cz \
    --target-lang en \
    --raw-data ~/shares/cz-en/data/raw \
    --dest-dir ./shared-bpe/data/prepared \
    --model-dir ./shared-bpe/tokenizers \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --vocab-size 16000 \


# TRAIN MODEL
python train.py \
    --cuda \
    --data shared-bpe/data/prepared/ \
    --src-tokenizer shared-bpe/tokenizers/cz-en-bpe-16000.model \
    --tgt-tokenizer shared-bpe/tokenizers/cz-en-bpe-16000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 64 \
    --arch transformer \
    --max-epoch 7 \
    --log-file shared-bpe/logs/train.log \
    --save-dir shared-bpe/checkpoints/ \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 300 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3 


# TRANSLATION
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer shared-bpe/tokenizers/cz-en-bpe-16000.model \
    --tgt-tokenizer shared-bpe/tokenizers/cz-en-bpe-16000.model \
    --checkpoint-path shared-bpe/checkpoints/checkpoint_best.pt \
    --output shared-bpe/output.txt \
    --max-len 300
