import argparse
import torch
import torch.distributed as dist
import os

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
    parser.add_argument('--max-iterations', type=int, default=10, help='How many iterations t? (default: 10)')
    parser.add_argument('--multiplier', type=str2bool, default=True, help='Use lag. multipliers?')
    parser.add_argument('--split', type=str2bool, default=True, help='split?')
    parser.add_argument('--partial', type=int, default=None, help='partial? (default: None)')
    args = parser.parse_args()

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
    # if use_cuda:
    #     xk_model = xk_model.to(cuda_device)
    #     yk_model = yk_model.to(cuda_device)
    x0_model = model.Net()
    # if use_cuda:
    #     x0_model = x0_model.to(cuda_device)

    dist.init_process_group(backend='gloo')
    print("init_process_group done")

    for t in range(args.max_iterations):

        # send out model
        if use_cuda:
            xk_model = xk_model.to(cpu_device)
            yk_model = yk_model.to(cpu_device)
        send(xk_model, dst=0, tag=1)
        if args.multiplier:
            send(yk_model, dst=0, tag=2)

        # receive x0 model
        if use_cuda:
            x0_model = x0_model.to(cpu_device)
        x0_model = receive(x0_model, src=0, tag=0)

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
