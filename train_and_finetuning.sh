CUDA_VISIBLE_DEVICES=0 python pretraining_eval_ver.py --neighbor_epoch 5 --l "20" --bert_layer "6" --hidden_size 64 --block_size 64 --mlm_prob 0.5 --input cora --position True --seed 44

CUDA_VISIBLE_DEVICES=0 python node_classification_task_simple.py --neighbor_epoch 5 --steps 0 --l "20" --bert_layer "6" --hidden_size 64 --block_size 64 --mlm_prob 0.5 --input cora --position True --seed 44

