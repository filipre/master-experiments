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
import random
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

    assert rank > 0, "Rank must not be 0"

    # TODO: remove unnec. arguments
    parser = argparse.ArgumentParser(description='ADMM Segmentation')
    parser.add_argument('--max-iterations', type=int, default=10000, help='How many iterations? (default: 10)')
    parser.add_argument('--scale-img-size', type=float, default=0.25, help='Rescale image')
    parser.add_argument('--k', type=int, default=4, help='Number of different segments')
    parser.add_argument('--image', type=str, default='butterfly.png', help='Image location')
    parser.add_argument('--tau', type=float, default=10, help='Tau')
    parser.add_argument('--delta', type=float, default=0.01, help='Huber Delta')
    parser.add_argument('--alpha', type=float, default=0.1, help='TV Alpha')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--random-sleep', type=int, default=3, help='rank depending sleep')
    parser.add_argument('--constant-sleep', type=int, default=0, help='rank depending sleep')
    args = parser.parse_args()

    cpu_device = torch.device("cpu")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        cuda_device = torch.device("cuda")
    else:
        cuda_device = None
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # TODO get splitted image data
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

    # important values for current node
    k = rank-1
    nk = n_k[k]
    Ak = A_k[k]
    Dk = D_k[k]

    # init data, will be overwritten later
    u0 = torch.rand(n, args.k)
    uk = torch.rand(n_k[k], args.k)
    pk = torch.rand(n_k[k], args.k)
    # TODO: send to GPU? .to(cuda_device)

    dist.init_process_group(backend='gloo')
    print("init_process_group done")

    for t in range(args.max_iterations):

        # TODO: switch between GPU and CPU for caluclating and receiving

        # send out model
        req_uk = dist.isend(tensor=uk, dst=0, tag=1)
        req_pk = dist.isend(tensor=pk, dst=0, tag=2)
        req_uk.wait()
        req_pk.wait()
        print("Model sent to master (rank 0)")

        # receive u0 model
        req = dist.irecv(tensor=u0, src=0, tag=0)
        req.wait() # TODO: verify
        # req = dist.irecv(tensor=u0, src=0, tag=0)
        # thr = threading.Thread(target=wait_thread, args=(req,), daemon=True)
        # thr.start()
        # for j in range(1000):
        #     print(f"waiting {j}/1000")
        #     if not thr.is_alive():
        #         break
        #     time.sleep(1)

        # primal update
        huber_D = Dk
        huber_v = torch.sparse.mm(Ak, u0) + (pk/args.tau)
        huber_L2 = 8 / nk
        uk = huberROF.solve(huber_D, huber_v, huber_L2, alpha=args.alpha, delta=args.delta, tau=args.tau, theta=1, verbose=False)

        # dual update
        pk = pk + args.tau * (torch.sparse.mm(Ak, u0) - uk)

        # random delay (max. 1min) to simulate network problems
        time.sleep(random.randint(0, args.random_sleep))
        time.sleep(args.constant_sleep)

# def wait_thread(req):
#     req.wait()

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
