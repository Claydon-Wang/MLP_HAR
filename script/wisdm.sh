export CUDA_VISIBLE_DEVICES=$1

# WISDM
MODELS=('cnn' 'mlp')

for MODEL in "${MODELS[@]}"; do
    python train_wisdm.py --model ${MODEL}
done
