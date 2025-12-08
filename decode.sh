#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=1:15:0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=assignment5/fast_decode5.out

module load miniforge3
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit


python translate.py \
    --input toy_example/data/raw/test.cz \
    --src-tokenizer shared-bpe/tokenizers/cz-en-bpe-16000.model \
    --tgt-tokenizer shared-bpe/tokenizers/cz-en-bpe-16000.model \
    --checkpoint-path shared-bpe/checkpoints/checkpoint_best.pt \
    --max-len 100 \
    --output assignment5/output_fast.txt \
    --bleu \
    --reference toy_example/data/raw/test.en \
    --cuda \
    --beam-size 5
