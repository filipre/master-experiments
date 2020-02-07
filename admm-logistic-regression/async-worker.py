import argparse
import torch
import torch.distributed as dist
import os
import time
import threading
import random

import dataloader
import model
import xkSolverNoMult
import xkSolverWithMult
import ykSolver

def main():

    assert "WORLD_SIZE" in  os.environ, "WORLD_SIZE not set"
    assert "RANK" in  os.environ, "RANK not set"
    assert "MASTER_ADDR" in  os.environ, "MASTER_ADDR not set"
    assert "MASTER_PORT" in  os.environ, "MASTER_PORT not set"

    world_size = int(os.environ['WORLD_SIZE'])
    number_nodes = world_size - 1
    rank = int(os.environ['RANK'])

    assert rank > 0, "Rank must not be 0"

    parser = argparse.ArgumentParser(description='Dist No Mult Worker')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--node-batch-size', type=int, default=64, metavar='N', help='input batch size within node (default: 64)')
    parser.add_argument('--node-epoch', type=int, default=1, metavar='N', help='number of epoch in node (default: 1)')
    parser.add_argument('--rho', type=float, default=1, help='Rho for node (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for node (default: 0.001)')
    parser.add_argument('--max-iterations', type=int, default=1000, help='How many iterations t? (default: 10)')
    parser.add_argument('--multiplier', type=str2bool, default=False, help='Use lag. multipliers?')
    parser.add_argument('--split', type=str2bool, default=False, help='split?')
    parser.add_argument('--partial', type=int, default=None, help='partial? (default: None)')
    parser.add_argument('--random-sleep', type=int, default=0, help='rank depending sleep')
    parser.add_argument('--constant-sleep', type=int, default=0, help='rank depending sleep')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    cpu_device = torch.device("cpu")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        cuda_device = torch.device("cuda")
    else:
        cuda_device = None
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.split:
        train_dataloader = dataloader.getSplittedTrianingLoadersForRank(rank, number_nodes, args.node_batch_size, kwargs, partial=args.partial)
    else:
        train_dataloader = dataloader.getSameTrainingLoader(args.node_batch_size, kwargs, partial=args.partial)

    xk_model = model.Net()
    yk_model = model.Net()
    x0_model = model.Net()

    dist.init_process_group(backend='gloo')
    print("init_process_group done")

    for t in range(args.max_iterations):

        # send out model
        if use_cuda:
            xk_model = xk_model.to(cpu_device)
            yk_model = yk_model.to(cpu_device)
        xk_weights = model.save(xk_model) # list of tensors
        reqs = []
        for i, xk_weight in enumerate(xk_weights):
            req = dist.isend(tensor=xk_weight, dst=0, tag=1*1000+i)
            reqs.append(req)
            print(f"xk_weight {i} sending out to {0}. Tag: {1*1000+i}")
        if args.multiplier:
            yk_weights = model.save(yk_model) # list of tensors
            for i, yk_weight in enumerate(yk_weights):
                req = dist.isend(tensor=yk_weight, dst=0, tag=2*1000+i)
                reqs.append(req)
                print(f"yk_weight {i} sending out to {0}. Tag: {2*1000+i}")
        for req in reqs:
            req.wait()

        # receive x0 model
        if use_cuda:
            x0_model = x0_model.to(cpu_device)
        x0_weights = model.save(x0_model) # will be overwritten but has the right structure
        wait_threads = []
        for i in range(len(x0_weights)):
            req = dist.irecv(tensor=x0_weights[i], src=0, tag=0*1000+i)
            thr = threading.Thread(target=wait_thread, args=(req,), daemon=True)
            thr.start()
            wait_threads.append(thr)
        # async wait, check every second TODO
        for j in range(1000):
            print(f"waiting {j}/1000")
            done = all(not t.is_alive() for t in wait_threads)
            if done:
                break
            time.sleep(1)
        x0_model = model.load(x0_weights, x0_model)

        # xk (and yk) update step
        device = cpu_device
        if use_cuda:
            x0_model = x0_model.to(cuda_device)
            xk_model = xk_model.to(cuda_device)
            yk_model = yk_model.to(cuda_device)
            device = cuda_device
        if args.multiplier:
            xk_model, scores, losses, residuals = xkSolverWithMult.solve(xk_model, train_dataloader, device, x0_model, yk_model, args.rho, args.lr, args.node_epoch)
            yk_model = ykSolver.solve(yk_model, x0_model, xk_model, args.rho)
        else:
            xk_model, scores, losses, residuals = xkSolverNoMult.solve(xk_model, train_dataloader, device, x0_model, args.rho, args.lr, args.node_epoch)

        # random delay (max. 1min) to simulate network problems
        time.sleep(random.randint(0, args.random_sleep))
        time.sleep(args.constant_sleep)


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
