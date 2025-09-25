It is the code for LFSS (Learning from Sample Stability for Deep Clustering)

## Training
```bash
torchrun  --nproc_per_node=1 --master_port 4831 train.py
```
You may choose a save_dir to save checkpoints and logs. 

And you can adjust the experimental setup and hyperparameters in the main.py.

## Test
```bash
torchrun  --nproc_per_node=1 --master_port 4831 eval.py
```
please modify the specific save_dir and checkpoint for testing.

## Acknowlegement
torch_clustering: https://github.com/Hzzone/torch_clustering

## Citation
If this code is helpful, you are welcome to cite our paper (Learning from Sample Stability for Deep Clustering).
