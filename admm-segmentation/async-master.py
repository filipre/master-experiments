import argparse
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
import os
import socket
import sys
from skimage.io import imread
from skimage.transform import rescale # , resize, downscale_local_mean
from sklearn.cluster import KMeans
import torchvision.transforms.functional as TF
import numpy as np
import time
import threading

import sparse
import projSimplex
import huberROF
import split

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

    # TODO: remove unnec. arguments
    parser = argparse.ArgumentParser(description='ADMM Segmentation')
    parser.add_argument('--max-iterations', type=int, default=100, help='How many iterations? (default: 10)')
    parser.add_argument('--scale-img-size', type=float, default=0.25, help='Rescale image')
    parser.add_argument('--k', type=int, default=4, help='Number of different segments')
    parser.add_argument('--image', type=str, default='butterfly.png', help='Image location')
    parser.add_argument('--tau', type=float, default=10, help='Tau')
    parser.add_argument('--delta', type=float, default=0.01, help='Huber Delta')
    parser.add_argument('--alpha', type=float, default=0.1, help='TV Alpha')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--barrier', type=int, default=1, help='Partial Barrier')
    args = parser.parse_args()

    filename = f"async_{args.image}_t{args.tau}_a{args.alpha}_del{args.delta}"
    image_filename = f'images/image_{filename}.pdf'
    plot_filename = f'plots/plot_{filename}.pdf'
    augmented_file = open(f'data/augmented_{filename}.csv', 'w+')
    delay_file = open(f'data/delays_{filename}.csv', 'w+')
    print(filename)

    # do not use cuda to avoid unnec. sending between gpu and cpu
    # TODO: later, do use cuda!
    use_cuda = False # not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # device = torch.device("cpu")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # print(device)

    # TODO: receive rhos from workers
    rhos = [args.tau] * number_nodes

    # Image
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
    f = torch.from_numpy(f).float()
    D = sparse.gradient_operator(ny, nx, ny, nx)
    ny_k, nx_k, c_k, n_k, A_k, D_k = [], [], [], [], [], []
    starts, lengths = split.split(nx, number_nodes)
    for k in range(number_nodes):
        ny_k.append(ny)
        nx_k.append(lengths[k])
        c_k.append(3)
        n_k.append(ny_k[k] * nx_k[k])
        Ak = sparse.selection_matrix(n_k[k], n, ny*starts[k])
        A_k.append(Ak)
        Dk = sparse.gradient_operator(ny_k[k], nx_k[k], ny, nx)
        D_k.append(Dk)
    sum_AkTAk = torch.zeros(n)
    for k in range(number_nodes):
        h1 = 0 + starts[k]*ny
        h2 = n_k[k]
        h3 = n - (h1 + h2)
        zeros1 = torch.zeros(h1)
        ones = torch.ones(h2)
        zeros2 = torch.zeros(h3)
        sum_AkTAk = sum_AkTAk + torch.cat((zeros1, ones, zeros2), 0)
    inv_AkTAk = torch.reciprocal(sum_AkTAk)
    inv_AkTAk = inv_AkTAk.view(-1, 1).repeat(1, args.k)

    # init models. will be overwritten eventually
    # TODO: send to GPU!
    # .to(device)
    u0 = torch.rand(n, args.k)
    uks, pks = [], []
    for k in range(number_nodes):
        uk = torch.rand(n_k[k], args.k)
        uks.append(uk)
        pk = torch.rand(n_k[k], args.k)
        pks.append(pk)

    augmented_lagrangians = []

    dist.init_process_group(backend='gloo')
    print("init_process_group done")

    # start all receiving jobs at the beginning
    threads = [] # list of pairs of threads
    for w in range(1, world_size): # for k in range(number_nodes):
        uk_req = dist.irecv(tensor=uks[w-1], src=w, tag=1)
        uk_thr = threading.Thread(target=wait_thread, args=(uk_req,), daemon=True)
        uk_thr.start()
        pk_req = dist.irecv(tensor=pks[w-1], src=w, tag=2)
        pk_thr = threading.Thread(target=wait_thread, args=(pk_req,), daemon=True)
        pk_thr.start()
        threads.append( (uk_thr, pk_thr) )

    node_iterations = [0] * number_nodes

    tic = time.time()

    t = 0
    # while t < args.max_iterations:
    while True:

        # check status if something has been received
        iteration_done = []
        for k in range(number_nodes):
            if not threads[k][0].is_alive() and not threads[k][1].is_alive():
                iteration_done.append(k)
        if len(iteration_done) < args.barrier:
            print(f"Not enough nodes ready: {len(iteration_done)}/{args.barrier}. Sleep...")
            time.sleep(1)
            continue

        t = t + 1

        # if we did receive something, perform updates
        print(f"Perform x0 update using nodes {iteration_done}")
        for k in iteration_done:
            node_iterations[k] = node_iterations[k] + 1
        delay_file.write(', '.join(str(e) for e in iteration_done))
        delay_file.write("\r\n")
        print(node_iterations)

        # u0 update
        sum_AkTuk = torch.zeros_like(u0)
        sum_AkTpk = torch.zeros_like(u0)
        for k in range(number_nodes):
            sum_AkTuk = sum_AkTuk + torch.sparse.mm(A_k[k].t(), uks[k])
            sum_AkTpk = sum_AkTpk + torch.sparse.mm(A_k[k].t(), pks[k])
        proj_v = inv_AkTAk * (sum_AkTuk - (sum_AkTpk + f)/args.tau)
        proj_H = torch.ones(args.k)
        u0 = projSimplex.scaled_proj(proj_v, proj_H, device)

        # send out new u0 model to iteration_done
        reqs = []
        for k in iteration_done:
            req = dist.isend(tensor=u0, dst=k+1, tag=0)
            reqs.append(req)

        # wait until everything has been transfered. TODO verify
        for req in reqs:
            req.wait()

        # start receiving from nodes again for iteration_done
        for k in iteration_done:
            uk_req = dist.irecv(tensor=uks[k], src=k+1, tag=1)
            uk_thr = threading.Thread(target=wait_thread, args=(uk_req,), daemon=True)
            uk_thr.start()
            pk_req = dist.irecv(tensor=pks[k], src=k+1, tag=2)
            pk_thr = threading.Thread(target=wait_thread, args=(pk_req,), daemon=True)
            pk_thr.start()
            threads[k] = (uk_thr, pk_thr)

        # Evaluation (while we are sending stuff....)
        uf = torch.sum(u0 * f)
        huberDu = args.alpha * torch.sum(huber(torch.sparse.mm(D, u0), args.delta))
        augmented = uf + huberDu
        for k in range(number_nodes):
            scalar_product = torch.sum(torch.mm(pks[k].t(), torch.sparse.mm(A_k[k], u0) - uks[k]))
            augmented = augmented - scalar_product
        for k in range(number_nodes):
            tau_norm = args.tau/2 * torch.pow(torch.norm(torch.sparse.mm(A_k[k], u0) - uks[k]), 2)
            augmented = augmented + tau_norm
        augmented_lagrangians.append(augmented)
        augmented_file.write(f"{augmented}\r\n")
        print(f"[{t}] {augmented}")

        # stop condition
        if all(it > args.max_iterations for it in node_iterations):
            break


    toc = time.time()
    tic_toc = toc - tic

    print("Done. Creating Graphs. Time:", tic_toc)
    result = u0.numpy()
    classes = np.argmax(result, axis=1)
    segmented_image = np.reshape(C[classes, :], (ny, nx, 3), order='F')
    segmented_image_color = []
    for k in range(args.k):
        segmented_image_color.append( np.reshape(result[:,k], (ny, nx), order='F') )
    # save image
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
    ax[2+args.k].plot(augmented_lagrangians)
    ax[2+args.k].set_yscale('log')
    ax[2+args.k].set_title('Augmented Lag.')
    # ax[3+args.k].plot(diff_uks)
    # ax[3+args.k].plot(diff_pks)
    # ax[3+args.k].set_yscale('log')
    # ax[3+args.k].set_title('Optimality condition [l1 norm]')
    fig.savefig(plot_filename, bbox_inches='tight')

def huber(x, delta):
    return torch.where(torch.le(torch.abs(x), delta), torch.pow(x, 2)/(2*delta), torch.abs(x) - delta/2)

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
