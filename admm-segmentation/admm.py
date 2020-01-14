import argparse
from skimage.io import imread
from skimage.transform import rescale # , resize, downscale_local_mean
from sklearn.cluster import KMeans
import torchvision.transforms.functional as TF
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import time

import sparse
import projSimplex
import huberROF
import split
import delay


def huber(x, delta):
    return torch.where(torch.le(torch.abs(x), delta), torch.pow(x, 2)/(2*delta), torch.abs(x) - delta/2)


def main():
    parser = argparse.ArgumentParser(description='ADMM Segmentation')
    parser.add_argument('--max-iterations', type=int, default=100, help='How many iterations? (default: 10)')
    parser.add_argument('--scale-img-size', type=float, default=0.25, help='Rescale image')
    parser.add_argument('--k', type=int, default=4, help='Number of different segments')
    parser.add_argument('--image', type=str, default='butterfly.png', help='Image location')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes / splits')
    parser.add_argument('--tau', type=float, default=1., help='Tau')
    parser.add_argument('--delta', type=float, default=1., help='Huber Delta')
    parser.add_argument('--alpha', type=float, default=1., help='TV Alpha')
    parser.add_argument('--delay', type=int, default=1, help='Delay')
    parser.add_argument('--delay-method', type=str, default='constant', help='constant, uniform, ...')
    args = parser.parse_args()

    filename = f"admm_{args.image}_t{args.tau}_a{args.alpha}_del{args.delta}_n{args.nodes}_d{args.delay}_dm{args.delay_method}"
    image_filename = f'images/image_{filename}.pdf'
    plot_filename = f'plots/plot_{filename}.pdf'
    augmented_file = open(f'data/augmented_{filename}.csv', 'w+')
    print(filename)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(device)

    raw_image = imread(args.image) # Image.open(args.image)
    raw_image = rescale(raw_image, args.scale_img_size, anti_aliasing=True, multichannel=True)
    img = np.array(raw_image, dtype=np.float64) # dtype=np.float64
    ny, nx, c = img.shape
    n = ny * nx
    print(f"{ny} x {nx} = {n}")
    img_vec = img.reshape((n, c), order='F') # TODO!

    kmeans = KMeans(n_clusters=args.k, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0).fit(img_vec)
    C = kmeans.cluster_centers_
    f = np.zeros((ny, nx, args.k))
    for k in range(args.k):
        cluster = C[k,:]
        tiled = np.tile(cluster, (ny, nx, 1))
        diff_img_clusters = np.power(img - tiled, 2)
        f[:, :, k] = np.sqrt(np.sum(diff_img_clusters, axis=2))
    f = f.reshape((n, args.k), order='F')
    f = torch.from_numpy(f).float() # TODO: .to(device) ?

    D = sparse.gradient_operator(ny, nx, ny, nx).to(device)

    ny_k, nx_k, c_k, n_k, A_k, D_k = [], [], [], [], [], []
    starts, lengths = split.split(nx, args.nodes)
    for k in range(args.nodes):
        ny_k.append(ny)
        nx_k.append(lengths[k])
        c_k.append(3)
        n_k.append(ny_k[k] * nx_k[k])
        Ak = sparse.selection_matrix(n_k[k], n, ny*starts[k]).to(device)
        A_k.append(Ak)
        Dk = sparse.gradient_operator(ny_k[k], nx_k[k], ny, nx).to(device)
        D_k.append(Dk)

    sum_AkTAk = torch.zeros(n).to(device)
    for k in range(args.nodes):
        h1 = 0 + starts[k]*ny
        h2 = n_k[k]
        h3 = n - (h1 + h2)
        zeros1 = torch.zeros(h1).to(device)
        ones = torch.ones(h2).to(device)
        zeros2 = torch.zeros(h3).to(device)
        sum_AkTAk = sum_AkTAk + torch.cat((zeros1, ones, zeros2), 0)
    inv_AkTAk = torch.reciprocal(sum_AkTAk)
    inv_AkTAk = inv_AkTAk.view(-1, 1).repeat(1, args.k)

    u0 = torch.rand(n, args.k).to(device)
    u0_queue = deque([u0])
    uk_queues, pk_queues = [], []
    for k in range(args.nodes):
        uk = torch.rand(n_k[k], args.k).to(device)
        uk_queues.append(deque([uk]))
        pk = torch.rand(n_k[k], args.k).to(device)
        pk_queues.append(deque([pk]))

    optimalities, diff_uks, diff_pks, augmenteds = [], [], [], []

    tic = time.time()

    for t in range(args.max_iterations):
        print(t)

        # Master update u0
        uks = delay.forMaster(uk_queues, args.delay, args.delay_method)
        pks = delay.forMaster(pk_queues, args.delay, args.delay_method)

        sum_AkTuk = torch.zeros_like(u0).to(device)
        sum_AkTpk = torch.zeros_like(u0).to(device)
        for k in range(args.nodes):
            sum_AkTuk = sum_AkTuk + torch.sparse.mm(A_k[k].t(), uks[k])
            sum_AkTpk = sum_AkTpk + torch.sparse.mm(A_k[k].t(), pks[k])

        proj_v = inv_AkTAk * (sum_AkTuk - (sum_AkTpk + f)/args.tau)
        proj_H = torch.ones(args.k).to(device)
        u0 = projSimplex.scaled_proj(proj_v, proj_H)

        u0_queue.appendleft(u0)
        if len(u0_queue) > args.delay:
            u0_queue.pop()

        # Worker updates uk
        for k in range(args.nodes):
            print(f"[Node {k}]")

            u0 = delay.forWorker(u0_queue, args.delay, args.delay_method)

            huber_D = D_k[k]
            huber_v = torch.sparse.mm(A_k[k], u0) + (pk_queues[k][0]/args.tau)
            huber_L2 = 8 / n_k[k]
            uk = huberROF.solve(huber_D, huber_v, huber_L2, device, alpha=args.alpha, delta=args.delta, tau=args.tau, theta=1, verbose=False)

            pk = pk_queues[k][0] + args.tau * (torch.sparse.mm(A_k[k], u0) - uk)

            uk_queues[k].appendleft(uk)
            pk_queues[k].appendleft(pk)
            assert len(uk_queues[k]) == len(pk_queues[k]), "queue problem"
            if len(uk_queues[k]) > args.delay:
                uk_queues[k].pop()
                pk_queues[k].pop()

        # Evaluation
        uf = torch.sum(u0_queue[0] * f)
        huberDu = args.alpha * torch.sum(huber(torch.sparse.mm(D, u0_queue[0]), args.delta))
        augmented = uf + huberDu
        for k in range(args.nodes):
            scalar_product = torch.sum(torch.mm(pk_queues[k][0].t(), torch.sparse.mm(A_k[k], u0_queue[0]) - uk_queues[k][0]))
            augmented = augmented - scalar_product # TODO! verify
        for k in range(args.nodes):
            tau_norm = args.tau/2 * torch.pow(torch.norm(torch.sparse.mm(A_k[k], u0_queue[0]) - uk_queues[k][0]), 2)
            augmented = augmented + tau_norm
        augmenteds.append(augmented)
        augmented_file.write(f"{augmented}\r\n")
        print(f"[{t}] {augmented}")

    toc = time.time()
    tic_toc = toc - tic

    print("Time:", tic_toc)

    result = u0_queue[0].numpy()
    classes = np.argmax(result, axis=1)
    segmented_image = np.reshape(C[classes, :], (ny, nx, 3), order='F')

    segmented_image_color = []
    for k in range(args.k):
        segmented_image_color.append( np.reshape(result[:,k], (ny, nx), order='F') )

    fig, ax = plt.subplots(figsize=(10,10))
    ax.axis('off')
    ax.imshow(segmented_image)
    fig.savefig(image_filename, bbox_inches='tight')

    fig, ax = plt.subplots(3+args.k, figsize=(10//2,10*(3+args.k)//2))
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    ax[0].imshow(img)

    ax[1].axis('off')
    ax[1].set_title(f"tau={args.tau}, alpha={args.alpha}, delta={args.delta}")
    ax[1].imshow(segmented_image)

    for k in range(args.k):
        ax[2+k].axis('off')
        ax[2+k].set_title(f"Color {k}")
        ax[2+k].imshow(segmented_image_color[k], cmap='plasma', vmin=0, vmax=1)

    ax[2+args.k].plot(augmenteds)
    ax[2+args.k].set_yscale('log')
    ax[2+args.k].set_title('Augmented Lag.')

    # ax[3+args.k].plot(diff_uks)
    # ax[3+args.k].plot(diff_pks)
    # ax[3+args.k].set_yscale('log')
    # ax[3+args.k].set_title('Optimality condition [l1 norm]')

    fig.savefig(plot_filename, bbox_inches='tight')

if __name__ == '__main__':
    main()
