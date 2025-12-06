#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_a5_mc_3.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRANSLATE
python translate.py \
    --cuda \
    --input ~/data/a5/atmt_2025/Assignment\ 5/toy_example/data/raw/test.cz \
    --src-tokenizer ~/data/atmt_2025/cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer ~/data/atmt_2025/cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path ~/data/atmt_2025/cz-en/checkpoints/checkpoint_best.pt \
    --output ~/data/a5/atmt_2025/Assignment\ 5/output-a5-mc-3.txt \
    --max-len 300 \
    --bleu \
    --reference ~/data/a5/atmt_2025/Assignment\ 5/toy_example/data/raw/test.en \
    --beam-size 7 \
    --alpha 0.2 \
    --maximum_candidates \
    --mc_size 3
