python train_synth.py --dataset synthetic_graph_cls --model-name FactorGNN --num_factors 4 --num-hidden 32 --num-layers 2 --num-latent 4 --seed 0 --log-subdir run0001

python train_synth.py --dataset synthetic_graph_cls --model-name GCN --num_factors 4 --num-hidden 32 --num-layers 2 --seed 0 --log-subdir run0001

python train_synth.py --dataset synthetic_graph_cls --model-name GAT --num_factors 4 --num-hidden 4 --num-layers 2 --num-heads 8 --seed 0 --log-subdir run0001

python train_synth.py --dataset synthetic_graph_cls --model-name MLP --num_factors 4 --num-hidden 32 --num-layers 2 --num-heads 4 --seed 0 --log-subdir run0001

python train_synth.py --dataset synthetic_graph_cls --model-name DisenGCN --seed 0 --log-subdir run0001