import time
import click
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
from matrix_completion import train_model, process_data
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils import load_all_p



################################################################################
# Command line arguments
################################################################################
@click.command()
@click.option('--missing_rate', type=float, default=0.5)
@click.option('--rank', type=int, default=60)
@click.option('--lr', type=float, default=1e-4)
@click.option('--epochs', type=int, default=10000)
@click.option('--init', type=str, default='svd')
@click.option('--repeat', type=int, default=1)
@click.option('--lamb', type=float, default=5e-4)

def main(missing_rate, rank, lr, epochs, init, repeat, lamb):

    acc, _, _ = load_all_p()
    data_acc = np.array(acc, dtype=np.float32).transpose()

    data_acc = process_data(data_acc, threshold=1e-2)

    all_p_matrices = [ data_acc ]
    all_names = ['acc']
    print(f'missing rate: [{missing_rate}]')
    print(f'rank: [{rank}]')
    print(f'init: [{init}]')
    print(f'learning rate: [{lr}]')
    print(f'epoch: [{epochs}]')
    print(f'lambda: [{lamb}]')
    print(f'repeat: [{repeat}]')
    start_time = time.time()

    for p_matrix, name in zip(all_p_matrices, all_names):

        print(f'low-rank matrix completion for [{name}] ========================')
        M_true = np.array(p_matrix)
        # mask = get_missing_data(M_true, missing_rate)
        # M_obs = M_true * mask

        Ms_hat, masks = train_model(M_true, missing_rate=missing_rate, rank=rank, lr=lr, epochs=epochs, init=init, repeat=repeat, lamb=lamb)
        end_time = time.time()
        print(f'time: {end_time - start_time}')
        # evaluation =============================================================
        norm_errors = []
        mapes_1 = []
        mapes_0 = []
        for M_hat, mask in zip(Ms_hat, masks):
            norm_errors.append(np.linalg.norm((M_true - M_hat) * (1 - mask)) / np.linalg.norm(M_true * (1 - mask)))
            
            mapes_1.append(float(mape(M_true[mask == 1], M_hat[mask == 1])))
            mapes_0.append(float(mape(M_true[mask == 0], M_hat[mask == 0])))

        print(f"MAPE 1 Error: {np.mean(mapes_1):.4f}")
        print(f"MAPE 0 Error: {np.mean(mapes_0):.4f}")


if __name__ == "__main__":


    main()