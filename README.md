# Pruning from Scratch

official implementation of the paper [Pruning from Scratch](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-WangY.403.pdf)

## Requirements
- pytorch == 1.1.0
- torchvision == 0.2.2
- [apex](https://github.com/NVIDIA/apex) @ commit: 574fe24

## CIFAR10
- learning channel importance gates from randomly initialized weights
```bash
python script/learn_gates.py -a ARCH --gpu GPU_ID --seed SEED -s SPARSITY -e EXPANSION
```
where `ARCH` is network architecture type, 
`SPARSITY` is the sparsity ratio $r$ in regularization term,
`EXPANSION` is expansion channel number of initial conv layer.
- pruning based on channel gates
```bash
python script/prune_model.py -a ARCH --gpu GPU_ID --seed SEED -s SPARSITY -e EXPANSION -p RATIO
```
where `RATIO` is the pruned model MACs reduction ratio, larger ratio indicates more compact model.
- training pruned model from scratch
```bash
python script/train_pruned.py -a ARCH --gpu GPU_ID --seed SEED -s SPARSITY -e EXPANSION -p RATIO --budget_train
```
where `--budget_train` activates the budget training scheme (Scratch-B) proposed in 
[Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270), 
which trains the pruned model for the same amount of computation bud- get with the full model.
Empirically, this training scheme is crucial for improving the pruned model performance.

## ImageNet
- prepare imagenet dataset following the instructions in 
[link](https://github.com/pytorch/examples/tree/master/imagenet), 
which results in an imagenet folder with train and val sub-folders.
- generate image index by
```bash
python script/prepare_imagenet_list.py --data_dir IMAGENET_DATA_DIR/train --dump_path data/train_images_list.pkl
python scrtpt/prepare_imagenet_list.py --data_dir IMAGENET_DATA_DIR/val --dump_path data/val_images_list.pkl
```
- learning channel importance gates from randomly initialized weights
```bash
python script/learn_gates_imagenet.py -a ARCH --gpu GPU_ID -s SPARSITY -e EXPANSION -m MULTIPLIER
```
where `MULTIPLIER` is used to control the expansion of channel number on the backbone outputs,
while `EXPANSION` is used to enlarge the intermediate channel numbers in InvertedResidual and Bottleneck blocks.
- pruning based on channel gates
```bash
python script/prune_model_imagenet.py -a ARCH --gpu GPU_ID -s SPARSITY -e EXPANSION -m MULTIPLIER -p RATIO
```
- training pruned model from scratch (single node multiple gpus)
```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPU script/train_pruned_imagenet.py \
    -a ARCH -e EXPANSION -s SPARSITY -p RATIO -m MULTIPLIER \
    -b TRAIN_BATCH_SIZE --lr LR --wd WD --lr_scheduler SCHEDULER \
    --budget_train --label_smooth
```
where `SCHEDULER` is learning rate scheduler type, 'multistep' for ResNet50, 'cos' for MobileNets.

## Citation
```bibtex
@inproceedings{wang2020pruning,
    title={Pruning from Scratch},
    author={Wang, Yulong and Zhang, Xiaolu and Xie, Lingxi and Zhou, Jun and Su, Hang and Zhang, Bo and Hu, Xiaolin},
    booktitle={Proceedings of the 29th International Joint Conference on Artificial Intelligence},
    year={2020},
    publisher={AAAI Press},
    address={New York, USA}
}
```
