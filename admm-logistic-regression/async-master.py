import argparse
import torch
from collections import deque
import matplotlib.pyplot as plt
import torch.distributed as dist
import os
import socket
import sys
import time
import threading

import dataloader
import model
import x0SolverNoMult
import x0SolverWithMult
import augLagrangianNoMult
import augLagrangianWithMult


"""
Master
[10] -> [0, 2]
[11] -> [3]
[12] -> [1]
[13] -> [0]
[14] -> [0, 1, 2]

Worker
[100] -> [10]
[101] -> [13]
"""

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
    parser.add_argument('--multiplier', type=str2bool, default=False, help='Use lag. multipliers?')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for node (default: 0.001)')
    parser.add_argument('--split', type=str2bool, default=False, help='split?')
    parser.add_argument('--barrier', type=int, default=1, help='Partial Barrier')
    parser.add_argument('--experiment', type=str, default="admm", help='Experiment identifier')
    args = parser.parse_args()

    filename = f'async_{args.experiment}_mult{args.multiplier}_split{args.split}_b{args.barrier}_r{args.rho}_lr{str(args.lr)}_n{number_nodes}.pdf'
    loss_file = open(f'data/loss_{filename}.csv', 'w+')
    delay_file = open(f'data/delays_{filename}.csv', 'w+')
    time_file = open(f'data/time_{filename}.csv', 'w+')
    print(filename)

    torch.manual_seed(args.seed)

    # do not use cuda to avoid unnec. sending between gpu and cpu
    use_cuda = False # not args.no_cuda and torch.cuda.is_available()
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

    wait_threads_for_nodes = []
    node_xk_weights = [None] * number_nodes # xk_weights
    node_yk_weights = [None] * number_nodes # yk_weights
    for k in range(number_nodes):
        node_xk_weights[k] = model.save(xk_models[k])
        node_yk_weights[k] = model.save(yk_models[k])

    # start all receiving jobs at the beginning
    for w in range(1, world_size): # for k in range(number_nodes):
        wait_threads_for_weights = []
        for i in range(len(node_xk_weights[w-1])):
            req = dist.irecv(tensor=node_xk_weights[w-1][i], src=w, tag=1*1000+i)
            thr = threading.Thread(target=wait_thread, args=(req,), daemon=True)
            thr.start()
            wait_threads_for_weights.append(thr)
        if args.multiplier:
            for i in range(len(node_yk_weights[w-1])):
                req = dist.irecv(tensor=node_yk_weights[w-1][i], src=w, tag=2*1000+i)
                thr = threading.Thread(target=wait_thread, args=(req,), daemon=True)
                thr.start()
                wait_threads_for_weights.append(thr)
        wait_threads_for_nodes.append(wait_threads_for_weights)

    node_iterations = [0] * number_nodes

    tic = time.time()
    t = 0
    while t < args.max_iterations:
        # check status if something has been received
        iteration_done = []
        for k in range(number_nodes):
            done = all(not thr.is_alive() for thr in wait_threads_for_nodes[k])
            if done:
                iteration_done.append(k)

        if len(iteration_done) < args.barrier:
            print(f"Not enough nodes ready: {len(iteration_done)}/{args.barrier}. Sleep...")
            time.sleep(1)
            continue

        print(f"Perform x0 update using nodes {iteration_done}")
        for k in iteration_done:
            node_iterations[k] = node_iterations[k] + 1
        delay_file.write(', '.join(str(e) for e in iteration_done))
        delay_file.write("\r\n")
        print(node_iterations)

        for k in iteration_done:
            xk_models[k] = model.load(node_xk_weights[k], xk_models[k])
            if args.multiplier:
                yk_models[k] = model.load(node_yk_weights[k], yk_models[k])

        # x0 update
        t = t + 1
        if args.multiplier:
            x0_model = x0SolverWithMult.solve(x0_model, xk_models, yk_models, rhos)
        else:
            x0_model = x0SolverNoMult.solve(x0_model, xk_models, rhos)

        # send out new x0 model to iteration_done
        x0_weights = model.save(x0_model)
        reqs = []
        for k in iteration_done:
            for i, x0_weight in enumerate(x0_weights):
                req = dist.isend(tensor=x0_weight, dst=k+1, tag=0*1000+i)
                print(f"x0_weight {i} sending out to {k+1}. Tag: {0*1000+i}")
                reqs.append(req)
        # TODO: wait?
        for req in reqs:
            req.wait()

        # start receiving from nodes again for iteration_done
        for k in iteration_done:
            node_xk_weights[k] = model.save(xk_models[k])
            node_yk_weights[k] = model.save(yk_models[k])
            wait_threads_for_weights = []
            for i in range(len(node_xk_weights[k])):
                req = dist.irecv(tensor=node_xk_weights[k][i], src=k+1, tag=1*1000+i)
                thr = threading.Thread(target=wait_thread, args=(req,), daemon=True)
                thr.start()
                wait_threads_for_weights.append(thr)
            if args.multiplier:
                for i in range(len(node_yk_weights[k])):
                    req = dist.irecv(tensor=node_yk_weights[k][i], src=k+1, tag=2*1000+i)
                    thr = threading.Thread(target=wait_thread, args=(req,), daemon=True)
                    thr.start()
                    wait_threads_for_weights.append(thr)
            wait_threads_for_nodes[k] = wait_threads_for_weights

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
        loss_file.write(f"{time.time() - tic}; {progress_loss}\r\n")
        print(f"Augmented Lagrangian: {aug_lagrangian}, Loss: {progress_loss}, Acc: {(progress_acc * 100):.1f}%")
        x0_model.train()
        for k in range(number_nodes):
            xk_models[k].train()
            yk_models[k].train()

        # stop condition
        if all(it > args.max_iterations for it in node_iterations):
            break

    toc = time.time()
    tic_toc = toc - tic
    time_file.write(str(tic_toc))

    print("DONE")

    # Create graphs
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

def wait_thread(req):
    req.wait()

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
