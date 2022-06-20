import numpy as np
import pandas as pd
import os, time, sys
import matplotlib.pyplot as plt


def coefficient(len, orders):
    # 计算系数
    bino = np.ndarray([len, orders.shape[0]])
    bino[0] = 1
    for i in range(1, len):
        bino[i] = (1 - (1 + orders) / i) * bino[i - 1]
    return bino


def FractionalKF(init_val, init_var, orders, n_iter, system, sys_inputs, ob, data, sys_var, ob_var):
    """system 应为一个列表，每个元素是一个函数，传入上一个状态和系统输入，返回一对值，第一个是下一个状态，第二个是切线性算子。 ob同理"""
    analysis_vars = []
    forecast_vars = []

    # 计算系数
    bino = coefficient(n_iter+1, orders)
    bino = bino[1:]

    forecast_state = init_val
    forecast_var = init_var
    forecast_states = forecast_state.copy().T
    forecast_vars.append(forecast_var)
    analysis_states = np.zeros([1, forecast_state.shape[0]])

    L = 50
    N = 50

    # 迭代
    for idx in range(n_iter):
        forecast_ob, ob_op = ob[idx](forecast_state)
        gain = forecast_var.dot(ob_op.T)\
            .dot(np.linalg.inv(
            ob_op.dot(forecast_var).dot(ob_op.T) + ob_var[idx]
            ))
        analysis_state = forecast_state + gain.dot(data[idx] - forecast_ob)
        analysis_var = (np.identity(gain.shape[0]) - gain.dot(ob_op)).dot(forecast_var)

        analysis_states = np.concatenate([analysis_states, analysis_state.T], axis=0)
        analysis_vars.append(analysis_var)

        # 预测步
        frac_diff, linear_system_op = system[idx](analysis_state, sys_inputs[idx])
        temp = analysis_states[-1:0:-1]
        l = temp.shape[0] if temp.shape[0] < L else L
        forecast_state = frac_diff - np.sum(bino[:l] * temp[:l], axis=0, keepdims=True).T

        # 计算预测方差，有点麻烦
        forecast_var = (linear_system_op + np.identity(forecast_state.shape[0]) * bino[0]).dot(analysis_var).dot(
            (linear_system_op + np.identity(forecast_state.shape[0]) * bino[0]).T
            ) + sys_var[idx]
        for j in range(2, N+2):
            if j > len(analysis_vars):
                break
            gamma = np.identity(forecast_state.shape[0]) * bino[j-1]
            forecast_var += gamma.dot(analysis_vars[-j]).dot(gamma.T)

        forecast_states = np.concatenate([forecast_states, forecast_state.T], axis=0)
        forecast_vars.append(forecast_var)

    # 保存结果
    np.savez("forecast_200.npz", forecast=forecast_states)
    np.savez("analysis_200.npz", analysis=analysis_states[1:])

    return forecast_states, analysis_states[1:]


def example1():
    def dynamic(state, input):
        A = np.array([[0.,1.], [-0.1,-0.2]])
        B = np.array([[0.], [1.]])
        return A.dot(state)+B.dot(input), A

    def observe(state):
        C = np.array([[0.1,0.3]])
        return C.dot(state), C

    def generate_data():
        x0 = np.array([[0.], [0.]])
        real_vals = x0.T.copy()
        input1 = np.array([[1.]])
        input2 = np.array([[-0.]])
        system_inputs = [input1 for _ in range(100)] + [input2 for _ in range(30)] \
                        + [input1 for _ in range(100)] + [input2 for _ in range(170)]

        orders = np.array([0.7, 1.2])
        A = np.array([[0., 1.], [-0.1, -0.2]])
        B = np.array([[0.], [1.]])
        C = np.array([[0.1, 0.3]])
        bino = coefficient(500, orders)[1:]

        # np.random.seed(42)
        ob = C.dot(x0) + np.random.normal(0,0.3)

        for i in range(1,399+1):
            diff = A.dot(x0) + B.dot(system_inputs[i-1]) + np.random.multivariate_normal([0,0], 0.003 * np.identity(2)).reshape([2,1])
            temp = real_vals[::-1]
            x0 = diff - np.sum(bino[:temp.shape[0]] * temp, axis=0, keepdims=True).T
            real_vals = np.concatenate([real_vals, x0.T], axis=0)

            temp = C.dot(x0) + np.random.normal(0, 0.3)
            ob = np.concatenate([ob,temp], axis=0)

        np.savez("./320KB/example1/real_vals_200.npz", x1=real_vals[:,0], x2=real_vals[:,1])
        np.savez("./320KB/example1/ob_200.npz", ob=ob)

        plt.plot(range(real_vals.shape[0]), real_vals)
        plt.show()

        # print(ob.shape)
        return ob, real_vals

    init = np.array([[0.], [0.]])
    init_var = 100 * np.identity(2)
    orders = np.array([0.7, 1.2])
    n_iter = 400
    system = [dynamic for _ in range(n_iter)]
    input1 = np.array([[1.]])
    input2 = np.array([[0.]])
    system_inputs = [input1 for _ in range(100)] + [input2 for _ in range(30)] \
                    + [input1 for _ in range(100)] + [input2 for _ in range(170)]
    ob = [observe for _ in range(n_iter)]
    data, ref = generate_data()
    data = [data[i].reshape([1,1]) for i in range(data.shape[0])]
    sys_var_ = 0.003 * np.identity(2)
    sys_var = [sys_var_ for _ in range(n_iter)]
    ob_var_ = np.array([[0.3]])
    ob_var = [ob_var_ for _ in range(n_iter)]
    forecast, analysis = FractionalKF(init_val=init, init_var=init_var, orders=orders, n_iter=n_iter, system=system,
                                      sys_inputs=system_inputs, ob=ob, data=data, sys_var=sys_var, ob_var=ob_var)

    plt.plot(range(n_iter), ref)
    plt.plot(range(n_iter), analysis)
    plt.show()
    plt.plot(range(n_iter), ref-analysis)
    plt.show()


def example2():
    def dynamic(state, input):
        output = np.ones([3,1], dtype=np.float64)
        output[0] = state[1]
        output[1] = -0.1 * state[0] - state[2] * state[1] + input[0]
        output[2] = 0
        A = np.zeros([3,3], dtype=np.float64)
        A[0,1] = 1
        A[1,0] = -0.1
        A[1,1] = -state[2]
        A[1,2] = -state[1]
        return output, A

    def observe(state):
        C = np.array([[0.1,0.3,0.]])
        return C.dot(state), C

    def generate_data():
        x0 = np.array([[0.], [0.]])
        real_vals = x0.T.copy()
        input1 = np.array([[1.]])
        input2 = np.array([[-1.]])
        system_inputs = [input1 for _ in range(50)] + [input2 for _ in range(50)] \
                        + [input1 for _ in range(50)] + [input2 for _ in range(50)]

        orders = np.array([0.7, 1.2])
        A = np.array([[0., 1.], [-0.1, -0.2]])
        B = np.array([[0.], [1.]])
        C = np.array([[0.1, 0.3]])
        bino = coefficient(200, orders)[1:]

        # np.random.seed(42)
        ob = C.dot(x0) + np.random.normal(0,0.3)

        for i in range(1,199+1):
            diff = A.dot(x0) + B.dot(system_inputs[i-1]) # + np.random.multivariate_normal([0,0], 0.3 * np.identity(2)).reshape([2,1])
            temp = real_vals[::-1]
            x0 = diff - np.sum(bino[:temp.shape[0]] * temp, axis=0, keepdims=True).T
            real_vals = np.concatenate([real_vals, x0.T], axis=0)

            temp = C.dot(x0) + np.random.normal(0, 0.3)
            ob = np.concatenate([ob,temp], axis=0)

        np.savez("./320KB/example2/real_vals_200.npz", x1=real_vals[:,0], x2=real_vals[:,1])
        np.savez("./320KB/example2/ob_200.npz", ob=ob)

        plt.plot(range(real_vals.shape[0]), real_vals)
        plt.show()

        # print(ob.shape)
        return ob, real_vals

    init = np.array([[0.], [0.], [0.2]])
    init_var = 100 * np.identity(3)
    init_var[2,2] = 0.1
    orders = np.array([0.7, 1.2, 1.])
    n_iter = 200
    system = [dynamic for _ in range(n_iter)]
    input1 = np.array([[1.]])
    input2 = np.array([[-1.]])
    system_inputs = [input1 for _ in range(50)] + [input2 for _ in range(50)] \
                    + [input1 for _ in range(50)] + [input2 for _ in range(50)]
    ob = [observe for _ in range(n_iter)]
    data, ref = generate_data()
    data = [data[i].reshape([1,1]) for i in range(data.shape[0])]
    sys_var_ = 0.003 * np.identity(3)
    # sys_var_[2,2] = 0.003
    sys_var = [sys_var_ for _ in range(n_iter)]
    ob_var_ = np.array([[0.03]])
    ob_var = [ob_var_ for _ in range(n_iter)]
    forecast, analysis = FractionalKF(init_val=init, init_var=init_var, orders=orders, n_iter=n_iter, system=system,
                                      sys_inputs=system_inputs, ob=ob, data=data, sys_var=sys_var, ob_var=ob_var)

    plt.plot(range(n_iter), ref)
    plt.plot(range(n_iter), analysis[:,:2])
    plt.show()
    # print(analysis[:,2])
    plt.plot(range(n_iter), np.expand_dims(analysis[:,2],axis=1))
    plt.show()
    plt.plot(range(n_iter), ref-analysis[:,:2])
    plt.show()



if __name__=='__main__':
    # print(np.random.get_state())
    example1()