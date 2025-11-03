#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment3.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRANSLATE
python translate.py \
    --cuda \
    --input ~/data/a3/atmt_2025/Assignment\ 3/toy_example/data/raw/test.cz \
    --src-tokenizer ~/data/atmt_2025/cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer ~/data/atmt_2025/cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path ~/data/atmt_2025/cz-en/checkpoints/checkpoint_best.pt \
    --output ~/data/a3/atmt_2025/Assignment\ 3/output-a3.txt \
    --max-len 300 \
    --beam-size 5