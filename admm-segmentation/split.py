from scipy import sparse

def split(nx, nodes):
    starts = [0]
    lengths = []

    integer, remainder = divmod(nx, nodes) # nx // nodes, nx % nodes

    for i in range(nodes):
        lengths.append(integer)
    for i in range(remainder):
        lengths[i] = lengths[i] + 1

    for i in range(nodes - 1):
        starts.append(starts[i] + lengths[i])

    for i in range(nodes - 1):
        lengths[i] = lengths[i] + 1 # create overlap except last one

    return starts, lengths


if __name__ == '__main__':
    nx = 120
    nodes = 3
    starts, lengths = split(100, nx, nodes)
    print(f"{nx} split by {nodes}:")
    print("Starts: ", starts)
    print("Lengths: ", lengths)
