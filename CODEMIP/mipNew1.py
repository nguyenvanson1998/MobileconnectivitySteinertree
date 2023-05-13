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
    D = R*math.sqrt(2)
    Uw = math.ceil(W/D)
    Uh = math.ceil(H/D)

    # def pos_to_cid(p: tuple):
    #     cx = math.floor(p[0] / D)
    #     cy = math.floor(p[1] / D)
    #     return cx + Uw * cy

    def cid_to_center(cid: int):
        cy = cid // Uw
        cx = cid - Uw * cy

        lx = cx * D
        rx = min((cx + 1) * D, W)
        ly = cy * D
        ry = min((cy + 1) * D, H)
        return tuple([(lx + rx) / 2, (ly + ry) / 2])

    # BS_cid = pos_to_cid(BS)
    # # cycles of each cell
    # t = []
    # for _ in range(Uw * Uh):
    #     t.append(set())
    # for car in cars:
    #     t[pos_to_cid(car)].add(car[3] * M + car[2])
    # "center" point of each cell
    points = []
    for cid in range(Uw * Uh):
        points.append(cid_to_center(cid))
    points.append(BS)
    points.extend(cars)
    # print(points)

    def connectable(cid1: int, cid2: int):
        if cid1 == cid2:
            return False
        if min(cid1, cid2) > Uw*Uh and points[cid1][3] != points[cid2][3]:
            return False
        p1 = points[cid1]
        p2 = points[cid2]
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 <= 4*R**2 + 0.001

    def get_edges(cid: int):
        edge_nodes = []
        for i in range(len(points)):
            if i != cid and connectable(cid, i):
                edge_nodes.append(i)
        return edge_nodes

    # print(t)
    edges = []
    for i in range(len(points)):
        edges.append(get_edges(i))
    # print(edges)
    # print(Uw*Uh)

    # build model
    solver = pywraplp.Solver.CreateSolver('SCIP')
    f = np.empty([len(points), len(points), M, K], dtype=object)
    for i in range(len(points)):
        for j in range(len(points)):
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
            for j in range(len(points)):
                c = None
                if j == Uw*Uh:
                    c = solver.Constraint(-1, -1)
                elif j == Uw*Uh + 1 + M*k + m:
                    c = solver.Constraint(1, 1)
                else:
                    c = solver.Constraint(0, 0)
                for i in edges[j]:
                    c.SetCoefficient(f[i][j][m][k], 1)
                    c.SetCoefficient(f[j][i][m][k], -1)

    # for k in range(K):
    #     for j in range(len(points)):
    #         for m in range(M):
    #             for i in range(len(points)):
    #                 if not connectable(i, j):
    #                     c = solver.Constraint(0, 0)
    #                     c.SetCoefficient(f[i][j][m][k], 1)

    x = np.empty(Uw * Uh, dtype=object)
    for i in range(Uw*Uh):
        x[i] = solver.IntVar(0, 1, 'x%d' % i)
    for i in range(Uw*Uh):
        c = solver.Constraint(0, solver.infinity())
        c.SetCoefficient(x[i], len(points)*M*K)
        for j in edges[i]:
            for m in range(M):
                for k in range(K):
                    c.SetCoefficient(f[i][j][m][k], -1)

    obj = solver.Objective()
    for i in range(Uw*Uh):
        obj.SetCoefficient(x[i], 1)
    obj.SetMinimization()

    solver.set_time_limit(60000*30)
    status = solver.Solve()

    with open(out_path, 'w') as ff:
        ff.write(str(solver.wall_time()) + '\n')
        if status == solver.OPTIMAL:
            ff.write('OPTIMAL')
        elif status == solver.FEASIBLE:
            ff.write('FEASIBLE')
        else:
            ff.write('FAILED')
            return

        ff.write('\n' + str(round(obj.Value())))
        for i in range(Uw * Uh):
            # print(x[i].solution_value())
            if x[i].solution_value() == 1:
                ff.write('\n%f %f' % points[i])

        # for m in range(M):
        #     for k in range(K):
        #         for j in range(len(points)):
        #             for i in edges[j]:
        #                 if f[i][j][m][k].solution_value() == 1:
        #                     print('{} {} {} {} {}'.format(i, j, m, k, f[i][j][m][k].solution_value()))
    return round(obj.Value())


if __name__ == '__main__':
    for i in range(10,19):
        print("Test " +str(i) +" begin:")
        x = solve('./Testmip/Test'+str(i)+'.inp', './Testmip/Test'+str(i)+'.out')
        print("KQ = " + str(x) +"\n")
        print("Done Test" + str(i))
