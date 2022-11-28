import numpy as np
from IPython import embed
import scipy.optimize
import matplotlib.pyplot as plt


def model_function(a, tau1, tau2):
    # y = (1-np.exp(x*a/tau)) + np.exp(x*a/tau)
    y = a * (1 - np.exp(-(x/tau1)))*np.exp(-(x/tau2))
    return y


def compute_observed_data(a, tau1, tau2):
    a = a + (np.random.randn() * 0.1)
    tau1 = tau1 + (np.random.randn() * 0.1)
    tau2 = tau2 + (np.random.randn() * 0.1)
    y = model_function(a, tau1, tau2)
    noise = np.random.randn(len(y)) * 0.01
    y = y + noise
    return y, a, tau1, tau2


def error_function(param_list):
    # unpack the parameter list
    a, tau1, tau2 = param_list
    # run the model with the new parameters, returning the info we're interested in
    result = model_function(a, tau1, tau2)
    # return the sum of the squared errors
    return sum((result - data) ** 2)


if __name__ == '__main__':
    x = np.linspace(0, 2, 1000)
    # Model Parameters
    param_a = 1.5
    param_a_bounds = (0, 2)
    param_tau_1 = 0.1
    param_tau1_bounds = (0.01, 1)
    param_tau_2 = 8
    param_tau2_bounds = (5, 12)

    # Observed data
    data, o_a, o_tau1, o_tau2 = compute_observed_data(param_a, param_tau_1, param_tau_2)
    # error = error_function([param_a, param_tau], data)

    # Minimize Error
    res = scipy.optimize.minimize(
        error_function, [param_a, param_tau_1, param_tau_2],
        # method='L-BFGS-B',
        method='Nelder-Mead',
        bounds=[param_a_bounds, param_tau1_bounds, param_tau2_bounds]
    )
    print('')
    print(res)
    a_model, tau1_model, tau2_model = res.x
    model_data = model_function(a_model, tau1_model, tau2_model)
    print(f'True Params: {param_a}, {param_tau_1}, {param_tau_2}')
    print(f'Data: {o_a}, {o_tau1}, {o_tau2}')
    print(f'Model: {a_model}, {tau1_model}, {tau2_model}')
    print('')
    print(f'RMSE: {np.sqrt(res.fun / len(data))}')
    plt.plot(x, data, 'b')
    plt.plot(x, model_data, 'r')
    plt.show()
    embed()
    exit()