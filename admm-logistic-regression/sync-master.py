import argparse
import torch
from collections import deque
import matplotlib.pyplot as plt
import torch.distributed as dist
import os
import socket
import sys

import dataloader
import model
import x0SolverNoMult
import x0SolverWithMult
import augLagrangianNoMult
import augLagrangianWithMult

def main():

    assert "WORLD_SIZE" in  os.environ, "WORLD_SIZE not set"
    assert "RANK" in  os.environ, "RANK not set"
    assert "MASTER_ADDR" in  os.environ, "MASTER_ADDR not set"
    assert "MASTER_PORT" in  os.environ, "MASTER_PORT not set"

    world_size = int(os.environ['WORLD_SIZE'])
    number_nodes = world_size - 1
    rank = int(os.environ['RANK'])

    assert rank == 0, "Master does not have rank 0"
    assert world_size > 1, "World size is smaller than 2 (no workers)"

    print(socket.gethostbyname(socket.gethostname()))
    sys.stdout.flush()

    parser = argparse.ArgumentParser(description='Hong\'s ADMM')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--max-iterations', type=int, default=10, help='How many iterations t? (default: 10)')
    parser.add_argument('--rho', type=float, default=1, help='Rho for all nodes (default: 100)')
    parser.add_argument('--multiplier', type=str2bool, default=True, help='Use lag. multipliers?')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for node (default: 0.001)')
    parser.add_argument('--split', type=str2bool, default=True, help='split?')
    # parser.add_argument('--lambda1', type=float, default=0.01, help='lambda 1 (default: 0.01)')
    # parser.add_argument('--lambda2', type=float, default=0.02, help='lambda 2 (default: 0.02)')
    args = parser.parse_args()

    filename = f'sync_mult{args.multiplier}_split{args.split}_r{args.rho}_lr{str(args.lr)}_n{number_nodes}.pdf'
    print(filename)

    # do not use cuda to avoid unnec. sending between gpu and cpu
    use_cuda = False # not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_dataloader = dataloader.getTestLoader(kwargs)
    progress_dataloader = dataloader.getProgressLoader(kwargs)

    # TODO: receive rhos from workers
    rhos = [args.rho] * number_nodes

    x0_model = model.Net().to(device)
    xk_models, yk_models = [], []
    for k in range(number_nodes):
        xk_dummy = model.Net().to(device)
        xk_models.append(xk_dummy)
        yk_dummy = model.Net().to(device)
        yk_models.append(yk_dummy)

    augmented_lagrangians, progress_losses, progress_accs = [], [], []

    dist.init_process_group(backend='gloo')
    print("init_process_group done")

    for t in range(args.max_iterations):

        # recieve models from workers
        for w in range(1, world_size):
            xk_models[w-1] = receive(xk_models[w-1], src=w, tag=1)
            if args.multiplier:
                yk_models[w-1] = receive(yk_models[w-1], src=w, tag=2)

        # x0 update
        x0_model.train()
        if args.multiplier:
            x0_model = x0SolverWithMult.solve(x0_model, xk_models, yk_models, rhos)
        else:
            x0_model = x0SolverNoMult.solve(x0_model, xk_models, rhos)

        # evaluation
        x0_model.eval()
        for k in range(number_nodes):
            xk_models[k].eval()
            yk_models[k].eval()
        if args.multiplier:
            aug_lagrangian, progress_loss, progress_acc = augLagrangianWithMult.get(progress_dataloader, device, x0_model, xk_models, yk_models, rhos)
        else:
            aug_lagrangian, progress_loss, progress_acc = augLagrangianNoMult.get(progress_dataloader, device, x0_model, xk_models, rhos)
        augmented_lagrangians.append(aug_lagrangian)
        progress_losses.append(progress_loss)
        progress_accs.append(progress_acc)
        print(f"[{t}] Augmented Lagrangian: {aug_lagrangian}, Loss: {progress_loss}, Acc: {(progress_acc * 100):.1f}%")

        # send out x0 model to workers
        for w in range(1, world_size):
            send(x0_model, dst=w, tag=0)

    # Create graphs
    print("DONE")
    # fig, ax = plt.subplots(1, figsize=(10,5))
    # ax.set_title('Augmented Lagrangian')
    # ax.set_yscale('log')
    # ax.plot(augmented_lagrangians)
    # fig.savefig(f"graphs/sync_auglag_{filename}", bbox_inches='tight')
    #
    # fig, ax = plt.subplots(1, figsize=(10,5))
    # ax.set_title('x0 Cross Entropy Loss')
    # ax.set_yscale('log')
    # ax.plot(progress_losses)
    # fig.savefig(f"graphs/sync_xentrop_{filename}", bbox_inches='tight')

    # detailed graph
    fig, ax = plt.subplots(3, figsize=(10,20))
    ax[0].set_title('Augmented Lagrangian')
    ax[0].plot(augmented_lagrangians)
    ax[1].set_title('x0 Cross Entropy Loss')
    ax[1].plot(progress_losses)
    ax[2].set_title('Accuracy')
    ax[2].plot(progress_accs)
    # ax[3].set_title('Node Objective Function Scores')
    # for k in range(args.number_nodes):
    #     ax[3].plot(node_scores[k])
    # ax[4].set_title('Node Losses')
    # for k in range(args.number_nodes):
    #     ax[4].plot(node_losses[k])
    # ax[5].set_title('L1 Residuals')
    # for k in range(args.number_nodes):
    #     ax[5].plot(node_residuals[k])
    fig.savefig(f"graphs/{filename}", bbox_inches='tight')


def send(theModel, dst, tag):
    weights = model.save(theModel)
    for i in range(len(weights)):
        dist.send(tensor=weights[i], dst=dst, tag=tag*100+i)
    print("Model sent to", dst)

def receive(theModel, src, tag):
    weights = model.save(theModel) # will be overwritten
    for i in range(len(weights)):
        dist.recv(tensor=weights[i], src=src, tag=tag*100+i)
    theModel = model.load(weights, theModel)
    print("Model received from", src)
    return theModel

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    main()
