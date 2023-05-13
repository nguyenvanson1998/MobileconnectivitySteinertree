from ortools.linear_solver import pywraplp
import numpy as np
import math

eps = np.finfo(float).eps


def read_file(in_path: str):
    with open(in_path, 'r') as f:
        W, H = [float(x) for x in f.readline().split()]
        BS = tuple([float(x) for x in f.readline().split()])
        M = int(f.readline())
        R = float(f.readline())
        K = int(f.readline())
        cars = []
        for k in range(K):
            for m in range(M):
                cars.append(tuple([float(x) for x in f.readline().split()] + [m, k]))
               

        return W, H, BS, M, R, K, cars


def solve(in_path: str, out_path: str):
    W, H, BS, M, R, K, cars = read_file(in_path)
    D = 2 * R
    Uw = math.ceil(W/D)
    Uh = math.ceil(H/D)

    def pos_to_cid(p: tuple):
        cx = math.floor(p[0] / D)
        cy = math.floor(p[1] / D)
        return cx + Uw * cy

    def cid_to_center(cid: int):
        cy = cid // Uw
        cx = cid - Uw * cy

        lx = cx * D
        rx = min((cx + 1) * D, W)
        ly = cy * D
        ry = min((cy + 1) * D, H)
        return tuple([(lx + rx) / 2, (ly + ry) / 2])

    def connectable(cid1: int, cid2: int):
        p1 = cid_to_center(cid1)
        p2 = cid_to_center(cid2)
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 <= 4*R**2

    def get_edges(cid: int):
        edge_nodes = []
        for i in range(Uw * Uh):
            if i != cid and connectable(cid, i):
                edge_nodes.append(i)
        return edge_nodes

    def get_adjacent(cid: int):
        cy = cid // Uh
        cx = cid - Uh * cy
        adj_nodes = []
        if cx > 0:
            adj_nodes.append((cx - 1) + Uh * cy)
        if cy > 0:
            adj_nodes.append(cx + Uh * (cy - 1))
        if cx < Uw - 1:
            adj_nodes.append((cx + 1) + Uh * cy)
        if cy < Uh - 1:
            adj_nodes.append(cx + Uh * (cy + 1))

        return adj_nodes

    BS_cid = pos_to_cid(BS)
    t = []
    for _ in range(Uw * Uh):
        t.append(set())
    for car in cars:
        t[pos_to_cid(car)].add(car[3]*M+car[2])
    # print(t)
    edges = []
    for i in range(Uw * Uh):
        edges.append(get_edges(i))
    # print(edges)

    # build model
    solver = pywraplp.Solver.CreateSolver('SCIP')
    x = np.empty(Uw*Uh, dtype=object)
    for i in range(Uw*Uh):
        x[i] = solver.IntVar(0, 1, 'x%d' % i)
    f = np.empty([Uw*Uh, Uw*Uh, M, K], dtype=object)
    for i in range(Uw*Uh):
        for j in range(Uw*Uh):
            if i == j:
                continue
            for m in range(M):
                for k in range(K):
                    f[i][j][m][k] = solver.IntVar(0, 1, 'f%d_%d_%d_%d' % (i, j, m, k))

    # def add_equation(x1, x2, const):
    #     c = solver.Constraint(const, const)
    #     c.SetCoefficient(x1, 1)
    #     c.SetCoefficient(x2, -1)
    #     print(x1.name() + ' - ' + x2.name() + ' = ' + str(const))

    for m in range(M):
        for k in range(K):
            if pos_to_cid(cars[k*M+m]) == BS_cid:
                continue
            for j in range(Uw*Uh):
                c = None
                if j == BS_cid:
                    c = solver.Constraint(-1, -1)
                elif k * M + m in t[j]:
                    c = solver.Constraint(1, 1)
                else:
                    c = solver.Constraint(0, 0)
                for i in edges[j]:
                    c.SetCoefficient(f[i][j][m][k], 1)
                    c.SetCoefficient(f[j][i][m][k], -1)

    for k in range(K):
        for j in range(Uw*Uh):
            if len(t[j]) > 0 and j != BS_cid and all(kk//M != k for kk in t[j]):
                for m in range(M):
                    for i in edges[j]:
                        c = solver.Constraint(0, 0)
                        c.SetCoefficient(f[i][j][m][k], 1)

    for i in range(Uw*Uh):
        c = solver.Constraint(0, solver.infinity())
        c.SetCoefficient(x[i], Uw*Uh*M*K)
        for j in edges[i]:
            for m in range(M):
                if pos_to_cid(cars[m]) == BS_cid:
                    continue
                for k in range(K):
                    c.SetCoefficient(f[i][j][m][k], -1)

    obj = solver.Objective()
    for i in range(Uw*Uh):
        if i != BS_cid and len(t[i]) == 0:
            obj.SetCoefficient(x[i], 1)
    obj.SetMinimization()

    # solver.set_time_limit(60*1000)
    status = solver.Solve()

    with open(out_path, 'w') as f:
        f.write(str(solver.wall_time()) + '\n')
        if status == solver.OPTIMAL:
            f.write('OPTIMAL')
        elif status == solver.FEASIBLE:
            f.write('FEASIBLE')
        else:
            f.write('FAILED')
            return

        f.write('\n' + str(round(obj.Value())))
        for i in range(Uw * Uh):
            if i != BS_cid and len(t[i]) ==0 and x[i].solution_value() == 1:
                f.write('\n%f %f' % cid_to_center(i))


if __name__ == '__main__':
    #solve('./OldTest/Test3.inp', './OldTest/Test3.out')
    solve('./OldTest/extest.inp', './Test/extest.out')
