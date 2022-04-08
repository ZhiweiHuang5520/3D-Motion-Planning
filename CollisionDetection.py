import numpy as np
import main
import matplotlib.pyplot as plt; plt.ion()


def get_intersect(origin, direction, block):
    lower = block[0:3]
    upper = block[3:6]
    for ind, ele in enumerate(direction):
        if ele == 0:
            box_max = max(lower[ind], upper[ind])
            box_min = min(lower[ind], upper[ind])
            if origin[ind] > box_max or origin[ind] < box_min:
                return False

    t0, t1 = 0, 1
    for i in range(3):
        if direction[i] != 0:
            inv_dir = 1 / direction[i]
            t_near = (lower[i] - origin[i]) * inv_dir
            t_far = (upper[i] - origin[i]) * inv_dir
            if t_near > t_far:
                t_near, t_far = t_far, t_near
            t0 = max(t_near, t0)
            t1 = min(t_far, t1)
            if t0 > t1:
                return False

    return True


def collision_detector(path, blocks, edge):
    if edge:
        origin = path[0, :]
        direction = path[1, :] - path[0, :]
        for i in range(blocks.shape[0]):
            block = blocks[i,:]
            if get_intersect(origin, direction, block):
                return True

        return False
    else:
        for ind in range(path.shape[0]-1):
            new_edge = path[ind:ind+2,:]
            if collision_detector(new_edge, blocks, edge=True):
                return True

        return False


if __name__=="__main__":
    blocks = np.array([[0,0,0,1,1,1,120,120,120],
                       [3,3,3,4,4,4,120,120,120]])
    path = np.array([[0.,0.,1.],
                    [3, 2, 3.2],
                    [2,3,2]])

    start = np.array([0, 0, 0])
    goal = np.array([6, 6, 6])
    boundary = np.array([[0,0,0,6, 6, 6,120,120,120]])
    fig, ax, hb, hs, hg = main.draw_map(boundary, blocks, start, goal)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-')

    print("Collision? ",collision_detector(path, blocks, edge=False))




