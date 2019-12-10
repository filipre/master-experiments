import argparse
import torch
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt

import model
import dataloader
import delay
import solver

def main():
    # training batch 1, test batch 1000, epoch 10, lr 0.01, momentum 0.5
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--max-delay', type=int, default=1, metavar='D',
                        help='maximal gradient delay (default: 1, i.e. no delay)')
    parser.add_argument('--check-progress', type=int, default=1, help='Check progress every n iterations (default: 10)')
    parser.add_argument('--delay-method', type=str, default='constant', help='constant, uniform, ...')
    parser.add_argument('--partial', type=int, default=None, help='partial? (default: None)')
    # parser.add_argument('--lambda1', type=float, default=0.01, help='lambda 1 (default: 0.01)')
    # parser.add_argument('--lambda2', type=float, default=0.02, help='lambda 2 (default: 0.02)')
    args = parser.parse_args()
    print(args)

    filename = f'dm{args.delay_method}_d{args.max_delay}_lr{str(args.lr)}'
    graph_filename = f'graphs/graph_{filename}.pdf'
    loss_file = open(f'data/loss_{filename}.csv', 'w+')
    acc_file = open(f'data/acc_{filename}.csv', 'w+')
    print(filename)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # setting device on GPU if available, else CPU
    print('Using device:', device)
    print()

    train_dataloader = dataloader.getSameTrainingLoader(args.batch_size, kwargs, partial=args.partial)
    test_dataloader = dataloader.getTestLoader(kwargs)
    progress_dataloader = dataloader.getProgressLoader(kwargs)

    x_model = model.Net().to(device) # This is the "master" model on which we update the parameters
    x_model.train()
    x_model_queue = deque([x_model])

    progress_losses, progress_accs = [], []

    for epoch in range(args.epochs):

        # Training
        for batch_idx, (data, target) in enumerate(train_dataloader):

            # TRAIN step
            x_model = model.Net().to(device)
            x_model.train()
            model.copyWeights(x_model_queue[0], x_model) # get most recent parameters
            delayed_model = delay.delayModel(x_model_queue, args.max_delay, args.delay_method)
            delayed_model.train()

            x_model, loss = solver.solve(x_model, data, target, device, delayed_model, args.lr)

            x_model_queue.appendleft(x_model)
            if len(x_model_queue) > args.max_delay:
                x_model_queue.pop() # if we have more models than we want, remove the oldest

        # Evaluation after each epoch
        x_model_queue[0].eval()
        progress_loss = 0
        progress_correct = 0
        with torch.no_grad():
            for data, target in progress_dataloader:
                data, target = data.to(device), target.to(device)
                output = x_model_queue[0](data)
                progress_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                progress_correct += pred.eq(target.view_as(pred)).sum().item()
        progress_loss /= len(progress_dataloader.dataset)
        progress_losses.append(progress_loss)
        progress_acc = progress_correct / len(progress_dataloader.dataset)
        progress_accs.append(progress_acc)
        loss_file.write(f"{progress_loss}\r\n")
        acc_file.write(f"{progress_acc}\r\n")
        print(f"[{epoch}] Progress Loss: {progress_loss}, Acc: {(progress_acc * 100):.1f}%")

    fig, ax = plt.subplots(2, figsize=(10,20))
    ax[0].set_title('Cross-entropy loss')
    ax[0].plot(progress_losses)
    ax[1].set_title('Accuracy')
    ax[1].plot(progress_accs)
    fig.savefig(graph_filename, bbox_inches='tight')

if __name__ == '__main__':
    main()
