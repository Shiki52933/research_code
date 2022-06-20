import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root
from FAdvec_Diff import StochasticEnsembleKF, ObserveErrorMatrices, ObserveOperator, B_alpha


class mesh:
    def __init__(self, mesh_points: np.ndarray):
        """传入的mesh应含端点"""
        """这个网格类是否一般有效或者只对-1 1有效我没有仔细思考，反正不显式含-1、1"""
        self.mesh_points = mesh_points.copy()
        N = mesh_points.shape[-1] - 2
        self.integral = np.zeros([N, N])

        self.integral[0,0] = (self.mesh_points[2] - self.mesh_points[0]) / 2
        self.integral[0,1] = (self.mesh_points[2] - self.mesh_points[1]) / 6
        for i in range(1, self.integral.shape[0] - 1):
            self.integral[i,i-1] = (self.mesh_points[i+1] - self.mesh_points[i]) / 6
            self.integral[i,i] = (self.mesh_points[i+2] - self.mesh_points[i]) / 2
            self.integral[i,i+1] = (self.mesh_points[i+2] - self.mesh_points[i+1]) / 6
        self.integral[-1,-2] = (self.mesh_points[-2] - self.mesh_points[-3]) / 6
        self.integral[-1,-1] = (self.mesh_points[-1] - self.mesh_points[-3]) / 2

        self.integral_derivatives = np.zeros([N, N])

        self.integral_derivatives[0,0] = 1/(self.mesh_points[1] - self.mesh_points[0]) + 1/(self.mesh_points[2] - self.mesh_points[1])
        self.integral_derivatives[0,1] = -1/(self.mesh_points[2] - self.mesh_points[1])
        for i in range(1, self.integral_derivatives.shape[-1] -1 ):
            self.integral_derivatives[i,i-1] = -1/(self.mesh_points[i+1] - self.mesh_points[i])
            self.integral_derivatives[i,i] = 1/(self.mesh_points[i+1] - self.mesh_points[i]) + 1 / (self.mesh_points[i+2] - self.mesh_points[i+1])
            self.integral_derivatives[i,i+1] = -1/(self.mesh_points[i+2] - self.mesh_points[i+1])
        self.integral_derivatives[-1,-2] = -1 / (self.mesh_points[-2] - self.mesh_points[-3])
        self.integral_derivatives[-1,-1] = 1/(self.mesh_points[-1] - self.mesh_points[-2]) + 1 / (self.mesh_points[-2] - self.mesh_points[-3])


class Burgers:
    def __init__(self, alpha, v, _mesh: mesh, delta_t):
        self.alpha = alpha
        self._mesh = _mesh
        self.C_alpha_delta_t = delta_t ** (-alpha) / math.gamma(2-alpha)
        self.N = self._mesh.integral.shape[-1]
        self.v = v

    @staticmethod
    def nonlinear_solver(c_alpha_delta, f, v, _mesh: mesh):
        n = f.shape[-1]

        def problem(a: np.ndarray):
            equations = []

            equation = c_alpha_delta * (np.sum(a * _mesh.integral[:, 0])) + np.sum(f * _mesh.integral[:, 0]) + \
                       a[1] * (a[0] + a[1]) / 6 \
                       + v * (np.sum(a * _mesh.integral_derivatives[:, 0]))
            equations.append(equation)

            for i in range(1, a.shape[0] - 1):
                equation = c_alpha_delta * (np.sum(a * _mesh.integral[:, i])) + np.sum(f * _mesh.integral[:, i]) + \
                           -(a[i - 1] - a[i + 1]) * (a[i - 1] + a[i] + a[i + 1]) / 6 \
                           + v * (np.sum(a * _mesh.integral_derivatives[:, i]))
                equations.append(equation)

            equation = c_alpha_delta * (np.sum(a * _mesh.integral[:, -1])) + np.sum(f * _mesh.integral[:, -1]) + \
                       - a[-2] * (a[-2] + a[-1]) / 6 + v * (np.sum(a * _mesh.integral_derivatives[:, -1]))
            equations.append(equation)

            return equations

        sol = root(problem, np.zeros([n]))

        return sol.x

    def ForwardOp(self, ensemble: np.ndarray, idx, sys_var):
        print(idx)
        n = ensemble.shape[0]//self.N
        b_alpha = B_alpha(self.alpha, n+2)
        sols = []

        for j in range(ensemble.shape[-1]):
            # 约定时间近的在下面
            former_info = ensemble[:,j].reshape([n, self.N]).copy()
            former_info[0] *= -b_alpha[n-1]
            for t in range(1,n):
                former_info[0] += (b_alpha[t] - b_alpha[t-1]) * former_info[n-t]
            former_info = former_info[0]
            former_info *= self.C_alpha_delta_t

            sol = Burgers.nonlinear_solver(self.C_alpha_delta_t, former_info, self.v, self._mesh).squeeze()
            sols.append(sol)

        sols = np.array(sols)
        sols += np.random.multivariate_normal(np.zeros([sys_var.shape[0]]), sys_var, size=ensemble.shape[-1])
        sols = sols.T

        return np.concatenate([ensemble, sols], axis=0)


def init(input):
    return np.sin( np.pi * input )


def init2(input):
    res = input.copy()
    pos_idx = res>0
    neg_idx = res<=0
    res[pos_idx] = -1
    res[neg_idx] = 1
    return res


def ob_op(input):
    return input[-80::20]


def nonlinear_fem_test(alpha):
    x_mesh = np.linspace(0,2,401)
    u0 = init(x_mesh)[1:-1]
    u0 = u0.reshape([u0.shape[-1],1])
    delta_t = 0.001
    param_mesh = mesh(x_mesh)

    alpha = alpha
    bur = Burgers(alpha,0.01/np.pi, param_mesh, delta_t)

    u_new = u0.copy()
    for i in range(int(1/delta_t)):
        print(i)
        u_new = bur.ForwardOp(u_new,i,np.zeros([u0.shape[0],u0.shape[0]]))
        u0 = np.concatenate([u0, u_new[-u0.shape[0]:]], axis=1)

    u0 = u0
    np.savez(str(alpha)+"_burgers.npz", sol=u0.T)

    """
    x_mesh = x_mesh[1:-1]
    t_mesh = np.linspace(0, 1, int(1/delta_t)+1)
    t,x = np.meshgrid(t_mesh, x_mesh)
    # print(x.shape, t.shape, u0.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(t,x,u0)
    plt.show()
    """


def FracBurgersENKFTest(alpha):
    alpha = alpha
    read = np.load(str(alpha)+"_burgers.npz")
    sol = read['sol']
    zero_bdy1 = np.zeros([sol.shape[0], 1])
    zero_bdy2 = np.zeros([sol.shape[0], 1])
    sol = np.concatenate([zero_bdy1, sol, zero_bdy2], axis=1)

    # alpha = 1.0
    v = 0.01 / np.pi
    x_mesh = np.linspace(0, 2, 201)
    delta_t = 0.01
    param_mesh = mesh(x_mesh)

    """依照惯例，在[0,2]上沿x取4个点（不含边界），沿t取5个点，共20个点，已知边界为0"""
    ob = sol[::250,::80].copy()
    ob = ob[:,1:-1]
    ob += np.random.normal(0,0.01,size=ob.shape).squeeze()

    # 生成enkf需要的变量
    ob_list = []
    for i in range(4):
        ob_list += [ob[i].reshape([ob.shape[1], 1])] + [None] * 24
    ob_list += [ob[-1].reshape([ob.shape[1], 1])]

    # 生成观测误差矩阵
    iters = len(ob_list)
    ob_err = np.identity(ob.shape[1]) * 0.2
    ob_errs = [ob_err] * len(ob_list)
    ob_errors = ObserveErrorMatrices(ob_errs)

    # 观测算子
    special_ob_op = ObserveOperator(ob_op)

    # 系统误差
    sys_var = np.identity(x_mesh.shape[0]-2) * 0.0001
    sys_vars = [sys_var] * iters

    # 初始值
    init_ave = init(x_mesh)[1:-1]
    init_ave += np.random.normal(0,0.01,size=init_ave.shape).squeeze()
    # init_ave = np.zeros([x_mesh.shape[0]-2])
    init_var = 1 * np.identity(x_mesh.shape[0]-2)

    # 辅助类
    bur = Burgers(alpha, v, param_mesh, delta_t)

    # enkf
    N = 10
    fianl_sum = 0
    results_sum = 0
    for idx in range(1, N + 1):
        ob_errors = ObserveErrorMatrices(ob_errs)
        results = StochasticEnsembleKF(20, init_ave, init_var, ob_list, iters, ob_errors, special_ob_op, bur.ForwardOp, sys_vars)

        # 评价结果
        final = results[-1]
        final = final.reshape([final.shape[0] // (x_mesh.shape[0]-2) , (x_mesh.shape[0]-2) ])
        fianl_sum += final

        for i in range(len(results)):
            results[i] = results[i][-(x_mesh.shape[0]-2):].squeeze()
        results = np.array(results)
        results_sum += results

    np.savez("burgers_final.npz", final=final)
    np.savez("burgers_results.npz", results=results)



    final = fianl_sum / N
    results = results_sum / N

    ref1 = sol[::10,::2]
    ref1 = ref1[:,1:-1]
    # x = np.expand_dims(np.linspace(L, R, 501), axis=0)
    # t = np.expand_dims(np.linspace(0, T, 501), axis=1)
    # ref = x * x * t
    # print(np.max(np.abs(sol-ref)))
    # print(results.shape, final.shape, ref1.shape)

    abs_err = np.max(np.abs(results - ref1), axis=1)
    plt.plot(range(abs_err.shape[0]), abs_err)
    plt.show()
    abs_err = np.max(np.abs(final - ref1), axis=1)
    plt.plot(range(abs_err.shape[0]), abs_err)
    plt.show()

    t_mesh = np.linspace(0, 1, int(1 / delta_t) + 1)
    x, t = np.meshgrid(x_mesh[1:-1], t_mesh)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(t, x, final)
    plt.show()



def analysis():
    final = np.load("burgers_final.npz")['final']

    x_mesh = np.linspace(0, 2, 101)
    x_mesh = x_mesh[1:-1]
    delta_t = 0.01
    t_mesh = np.linspace(0, 1, int(1 / delta_t) + 1)
    x, t = np.meshgrid(x_mesh, t_mesh)
    print(x.shape, t.shape, final.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, t, final)
    plt.show()

    """
    read = np.load(str(1.0) + "_burgers.npz")
    sol = read['sol']
    zero_bdy1 = np.zeros([sol.shape[0], 1])
    zero_bdy2 = np.zeros([sol.shape[0], 1])
    sol = np.concatenate([zero_bdy1, sol, zero_bdy2], axis=1)

    ref1 = sol[::10, ::2]
    ref1 = ref1[:, 1:-1]
    """


def plot_frac_burgers():
    alphas = np.linspace(0.2,1.0,9)
    for i in range(alphas.shape[0]):
        print(alphas[i])
        # nonlinear_fem_test(alphas[i])


        x_mesh = np.linspace(0, 2, 401)
        delta_t = 0.001
        t_mesh = np.linspace(0, 1, int(1 / delta_t) + 1)
        x, t = np.meshgrid(x_mesh, t_mesh)
        # print(x.shape, t.shape, u0.shape)

        read = np.load(str(alphas[i]) + "_burgers.npz")
        sol = read['sol']
        zero_bdy1 = np.zeros([sol.shape[0], 1])
        zero_bdy2 = np.zeros([sol.shape[0], 1])
        sol = np.concatenate([zero_bdy1, sol, zero_bdy2], axis=1)
        # print(x.shape, t.shape, sol.shape)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(t, x, sol)
        plt.show()


if __name__=='__main__':
    FracBurgersENKFTest(0.8)