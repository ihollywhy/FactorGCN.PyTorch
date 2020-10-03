python train_zinc.py --dataset zinc --gpu 0 --model-name FactorGNN --dis-weight 0.2 --num-latent 8 --num-hidden 144 --seed 1 --log-subdir run0001

python train_zinc.py --dataset zinc --gpu 0 --model-name GCN --num-hidden 144 --num-layers 3 --seed 1 --log-subdir run0001

python train_zinc.py --dataset zinc --gpu 0 --model-name MLP --num-hidden 144 --num-layers 3 --seed 1 --log-subdir run0001

python train_zinc.py --dataset zinc --gpu 0 --model-name GAT --num-hidden 18 --num-heads 8 --num-layers 3 --seed 1 --log-subdir run0001

python train_zinc.py --dataset zinc --gpu 0 --model-name DisenGCN --seed 1 --log-subdir run0001