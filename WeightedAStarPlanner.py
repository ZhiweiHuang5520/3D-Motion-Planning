import numpy as np
import CollisionDetection as CD


def huristic_fun(curr, goal):
    return max(abs(curr - goal)) + 0.7*min(abs(curr - goal))


def cost(node, goal):
    return np.sqrt(sum((node - goal) ** 2))


class Planner:

    def __init__(self, boundary, blocks, resolution=0.1):
        self.boundary = boundary*10
        self.blocks = blocks*10
        self.resolution = resolution*10
        self.map_values_x = np.arange(self.boundary[0, 0], self.boundary[0, 3]+self.resolution,self.resolution).tolist()
        self.map_values_y = np.arange(self.boundary[0, 1], self.boundary[0, 4] + self.resolution, self.resolution).tolist()
        self.map_values_z = np.arange(self.boundary[0, 2], self.boundary[0, 5] + self.resolution, self.resolution).tolist()
        self.size_x = len(self.map_values_x)
        self.size_y = len(self.map_values_y)
        self.size_z = len(self.map_values_z)
        self.g_mat = np.ones((self.size_x, self.size_y, self.size_z)) * np.inf
        self.parent = np.empty_like((self.g_mat)).astype(object)


    def find_gvlaue(self, curr, change=None):
        # find or change node g value
        indx = self.map_values_x.index(curr[0])
        indy = self.map_values_y.index(curr[1])
        indz = self.map_values_z.index(curr[2])
        if change != None:
            self.g_mat[indx, indy, indz] = change
        return self.g_mat[indx, indy, indz]

    def find_parent(self, curr, change=None):
        # find or change node's parent
        indx = self.map_values_x.index(curr[0])
        indy = self.map_values_y.index(curr[1])
        indz = self.map_values_z.index(curr[2])
        if change != None:
            self.parent[indx, indy, indz] = change
        return self.parent[indx, indy, indz]

    def ind_map(self, curr):
        indx = self.map_values_x.index(curr[0])
        indy = self.map_values_y.index(curr[1])
        indz = self.map_values_z.index(curr[2])
        return indx,indy,indz


    def plan(self, start, goal, weight=1):
        astart = 10*start
        astart[0]+=min(abs(self.map_values_x-astart[0]))
        astart[1] += min(abs(self.map_values_y - astart[1]))
        astart[2] += min(abs(self.map_values_z - astart[2]))
        astart = astart.tolist()
        agoal = 10*goal
        agoal[0] += min(abs(self.map_values_x - agoal[0]))
        agoal[1] += min(abs(self.map_values_y - agoal[1]))
        agoal[2] += min(abs(self.map_values_z - agoal[2]))
        agoal = agoal.tolist()
        self.find_gvlaue(astart, change=0)
        open = [astart]
        closed = []
        path = [agoal]
        f_list = [self.find_gvlaue(astart) + weight * huristic_fun(start,goal)]
        [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        dR = np.delete(dR, 13, axis=1)
        dR = dR.transpose()*self.resolution
        while agoal not in closed:
            remo_ind = f_list.index(min(f_list))
            out_node = open[remo_ind]
            open.pop(remo_ind)
            f_list.pop(remo_ind)
            closed.append(out_node)

            # if len(closed)%1000 == 0:
            #     print("closed: ",len(closed))

            children = (dR+np.array(out_node)).tolist()
            for child in children:
                if child[0] not in self.map_values_x or child[1] not in self.map_values_y or child[2] not in self.map_values_z:
                    break

                new_edge = np.vstack((out_node, child))
                collision = CD.collision_detector(new_edge, self.blocks, edge=True)
                if child not in closed and (not collision):
                    cx,cy,cz = self.ind_map(child)
                    cost_v = cost(np.array(out_node)/10, np.array(child)/10)
                    g_cost = self.find_gvlaue(out_node, change=None) + cost_v
                    if self.g_mat[cx,cy,cz] > g_cost:
                        self.g_mat[cx,cy,cz] = g_cost
                        self.parent[cx,cy,cz] = out_node
                        if child in open:
                            f_ind = open.index(child)
                            f_list[f_ind] = g_cost + weight * huristic_fun(np.array(child)/10, goal)
                        else:
                            open.append(child)
                            f_list.append(self.g_mat[cx,cy,cz] + weight * huristic_fun(np.array(child)/10, goal))
        print("Finished: closed size", len(closed))
        # Recover path from parents
        while (astart not in path):
            child_node = path[-1]
            parent_node = self.find_parent(child_node)
            path.append(parent_node)
        path.reverse()
        path = (np.array(path)/10).tolist()

        if path[0] != start.tolist():
            path[0] = start.tolist()
        elif path[-1] != goal.tolist():
            path[-1] = goal.tolist()

        return np.array(path)
