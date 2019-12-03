import argparse
import torch
from collections import deque
import matplotlib.pyplot as plt

import dataloader
import model
import delay
import x0SolverNoMult
import x0SolverWithMult
import xkSolverNoMult
import xkSolverWithMult
import ykSolver
import augLagrangianNoMult
import augLagrangianWithMult

def main():
    parser = argparse.ArgumentParser(description='Hong\'s ADMM')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--max-delay', type=int, default=1, metavar='D',
                        help='maximal gradient delay (default: 1, i.e. no delay)')
    parser.add_argument('--number-nodes', type=int, default=3,
                        help='How many nodes should we simulate? (default: 10)')
    parser.add_argument('--node-batch-size', type=int, default=64, metavar='N',
                        help='input batch size within node (default: 64)')
    parser.add_argument('--node-epoch', type=int, default=1, metavar='N',
                        help='number of epoch in node (default: 1)')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='How many iterations t? (default: 10)')
    parser.add_argument('--rho', type=float, default=1,
                        help='Rho for all nodes (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for all nodes (default: 0.001)')
    parser.add_argument('--delay-method', type=str, default='constant', help='constant, uniform, ...')
    parser.add_argument('--multiplier', type=str2bool, default=False, help='Use lag. multipliers?')
    parser.add_argument('--split', type=str2bool, default=False, help='split?')
    parser.add_argument('--partial', type=int, default=None, help='partial? (default: None)')
    # parser.add_argument('--lambda1', type=float, default=0.01, help='lambda 1 (default: 0.01)')
    # parser.add_argument('--lambda2', type=float, default=0.02, help='lambda 2 (default: 0.02)')
    args = parser.parse_args()

    filename = f'dm{args.delay_method}_d{args.max_delay}_mult{args.multiplier}_split{args.split}_r{args.rho}_lr{str(args.lr)}_n{args.number_nodes}.pdf'
    print(filename)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # setting device on GPU if available, else CPU
    print('Using device:', device)
    print()

    if args.split:
        train_dataloader = dataloader.getSplittedTrainingLoaders(args.number_nodes, args.node_batch_size, kwargs, partial=args.partial)
    else:
        train_dataloader = dataloader.getSameTrainingLoaders(args.number_nodes, args.node_batch_size, kwargs, partial=args.partial)
    test_dataloader = dataloader.getTestLoader(kwargs)
    progress_dataloader = dataloader.getProgressLoader(kwargs)

    # Initialization
    x0_model = model.Net().to(device) #x0
    x0_model_queue = deque([x0_model])
    xk_models_queue, yk_models_queue = [], []
    for node in range(args.number_nodes):
        xk_model = model.Net().to(device) # model.copyWeights(x0_model_delays[0], x_model) # start with the same weights
        xk_models_queue.append( deque([xk_model]) )
        yk_model = model.Net().to(device)
        yk_models_queue.append( deque([yk_model]) )

    rhos = [args.rho] * args.number_nodes
    lrs = [args.lr] * args.number_nodes

    augmented_lagrangians, progress_losses, progress_accs = [], [], []
    node_scores, node_losses, node_residuals = [], [], []
    for node in range(args.number_nodes):
        node_scores.append([])
        node_losses.append([])
        node_residuals.append([])

    # Algorithm
    for t in range(args.max_iterations):

        x0_model = model.Net().to(device)
        x0_model.train()
        model.copyWeights(x0_model_queue[0], x0_model)
        xk_models = delay.forMaster(xk_models_queue, args.max_delay, args.delay_method)
        yk_models = delay.forMaster(yk_models_queue, args.max_delay, args.delay_method)

        # x0 update
        if args.multiplier:
            x0_model = x0SolverWithMult.solve(x0_model, xk_models, yk_models, rhos)
        else:
            x0_model = x0SolverNoMult.solve(x0_model, xk_models, rhos)

        # push new model to queue
        x0_model_queue.appendleft(x0_model)
        if len(x0_model_queue) > args.max_delay:
            x0_model_queue.pop() # remove oldest model

        for k in range(args.number_nodes):

            xk_model = model.Net().to(device)
            xk_model.train()
            model.copyWeights(xk_models_queue[k][0], xk_model)
            yk_model = model.Net().to(device)
            yk_model.train()
            model.copyWeights(yk_models_queue[k][0], yk_model)
            x0_model = delay.forWorker(x0_model_queue, args.max_delay, args.delay_method)

            # xk update
            if args.multiplier:
                xk_model, scores, losses, residuals = xkSolverWithMult.solve(xk_model, train_dataloader[k], device, x0_model, yk_model, rhos[k], lrs[k], args.node_epoch)
            else:
                xk_model, scores, losses, residuals = xkSolverNoMult.solve(xk_model, train_dataloader[k], device, x0_model, rhos[k], lrs[k], args.node_epoch)
            node_scores[k] = node_scores[k] + scores
            node_losses[k] = node_losses[k] + losses
            node_residuals[k] = node_residuals[k] + residuals

            # yk update
            if args.multiplier:
                yk_model = ykSolver.solve(yk_model, x0_model, xk_model, rhos[k])

            xk_models_queue[k].appendleft(xk_model)
            yk_models_queue[k].appendleft(yk_model)
            assert len(xk_models_queue[k]) == len(yk_models_queue[k]), "something is wrong wiht the queues"
            if len(xk_models_queue[k]) > args.max_delay:
                xk_models_queue[k].pop()
                yk_models_queue[k].pop()

        # evaluation
        x0_model = x0_model_queue[0]
        x0_model.eval()
        xk_models, yk_models = [], []
        for k in range(args.number_nodes):
            xk_model = xk_models_queue[k][0]
            xk_model.eval()
            xk_models.append(xk_model)
            yk_model = yk_models_queue[k][0]
            yk_model.eval()
            yk_models.append(yk_model)

        if args.multiplier:
            aug_lagrangian, progress_loss, progress_acc = augLagrangianWithMult.get(progress_dataloader, device, x0_model, xk_models, yk_models, rhos)
        else:
            aug_lagrangian, progress_loss, progress_acc = augLagrangianNoMult.get(progress_dataloader, device, x0_model, xk_models, rhos)
        augmented_lagrangians.append(aug_lagrangian)
        progress_losses.append(progress_loss)
        progress_accs.append(progress_acc)
        print(f"[{t}] Augmented Lagrangian: {aug_lagrangian}, Loss: {progress_loss}, Acc: {(progress_acc * 100):.1f}%")


    # Create graphs
    print("DONE")
    fig, ax = plt.subplots(1, figsize=(10,5))
    ax.set_title('Augmented Lagrangian')
    ax.set_yscale('log')
    ax.plot(augmented_lagrangians)
    fig.savefig(f"graphs/auglag_{filename}", bbox_inches='tight')

    fig, ax = plt.subplots(1, figsize=(10,5))
    ax.set_title('x0 Cross Entropy Loss')
    ax.set_yscale('log')
    ax.plot(progress_losses)
    fig.savefig(f"graphs/xentrop_{filename}", bbox_inches='tight')

    # detailed graph
    fig, ax = plt.subplots(6, figsize=(10,20))
    ax[0].set_title('Augmented Lagrangian')
    ax[0].plot(augmented_lagrangians)
    ax[1].set_title('x0 Cross Entropy Loss')
    ax[1].plot(progress_losses)
    ax[2].set_title('Accuracy')
    ax[2].plot(progress_accs)
    ax[3].set_title('Node Objective Function Scores')
    for k in range(args.number_nodes):
        ax[3].plot(node_scores[k])
    ax[4].set_title('Node Losses')
    for k in range(args.number_nodes):
        ax[4].plot(node_losses[k])
    ax[5].set_title('L1 Residuals')
    for k in range(args.number_nodes):
        ax[5].plot(node_residuals[k])
    fig.savefig(f"graphs/details_{filename}", bbox_inches='tight')

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
