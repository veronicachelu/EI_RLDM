import numpy as np
import scipy.optimize as opt  # root-finding algorithm
import importlib  # Built-in module
from functools import partial
from src import plotting
from src import utils
from src import tuning

plotting = importlib.reload(plotting)
utils = importlib.reload(utils)
tuning = importlib.reload(tuning)

def my_fp(tau_E, a_E, b_E, d_E, f_E, dimA_E,
          tau_I, a_I, b_I, d_I, f_I, dimA_I,
          wEE, wEI, wIE, wII,
          I_ext_E, I_ext_I,
          rE_init, rI_init,
          **other_pars):
    """
    Use opt.root function to solve Equations (2)-(3) from initial values
    """

    F_E = partial(f_E,  a=a_E, b=b_E, d=d_E)
    F_I = partial(f_I,  a=a_I, b=b_I, d=d_I)

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])
    W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])
    W_II = np.array([[1 + wII, 1 - wII], [1 - wII, 1 + wII]])

    # define the right hand of wilson-cowan equations
    def my_WCr(x):

        rE, rI = x

        rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
        rE_wEI = np.einsum("j,jk->k", rE, W_EI) if dimA_E > 1 else rE * wEI
        rI_wIE = np.einsum("j,jk->k", rI, W_IE) if dimA_I > 1 else rI * wIE
        rI_wII = np.einsum("j,jk->k", rI, W_II) if dimA_I > 1 else rI * wII

        drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E)) / tau_E
        drIdt = (-rI + F_I(rE_wEI - rI_wII + I_ext_I)) / tau_I
        y = np.array([drEdt, drIdt])

        return y

    x0 = np.array([rE_init, rI_init])
    x_fp = opt.root(my_WCr, x0).x

    return x_fp

def my_fp_E(tau_E, noise, a_E, b_E, d_E, f_E, dimA_E, dimA_I, wEE,
            wIE, I_ext_E, rE_init, rI_init, **other_pars):
    """
    Use opt.root function to solve Equations (2)-(3) from initial values
    """

    F_E = partial(f_E,  a=a_E, b=b_E, d=d_E)

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])

    # define the right hand of wilson-cowan equations
    def my_WCr(rE):

        rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
        rI_wIE = np.einsum("j,jk->k", rI_init, W_IE) if dimA_I > 1 else rI_init * wIE

        drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E + noise)) / tau_E
        y = drEdt

        return y

    x0 = np.array(rE_init)
    x_fp = opt.root(my_WCr, x0).x

    return x_fp

def my_fp_E2(tau_E, a_E, b_E, d_E, f_E, dimA_E,
            dimA_I, a_I, b_I, d_I, f_I,wEI,
            wEE, wIE,
            I_ext_E,
            rE_init, noise_mean,
            **other_pars):
    """
    Use opt.root function to solve Equations (2)-(3) from initial values
    """

    F_E = partial(f_E,  a=a_E, b=b_E, d=d_E)

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])

    # define the right hand of wilson-cowan equations
    def my_WCr(rE):

        rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
        W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])
        F_I = partial(f_I, a=a_I, b=b_I, d=d_I)
        rE_wEI = np.einsum("j,jk->k", rE, W_EI) if dimA_E > 1 else rE * wEI
        rI = F_I(rE_wEI)

        rI_wIE = np.einsum("j,jk->k", rI, W_IE) if dimA_I > 1 else rI * wIE

        drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E + noise_mean)) / tau_E
        y = drEdt

        return y

    x0 = np.array(rE_init)
    x_fp = opt.root(my_WCr, x0).x

    return x_fp


def check_fp(pars, x_fp, mytol=1e-6):
    """
    Verify (drE/dt)^2 + (drI/dt)^2< mytol

    Args:
      pars    : Parameter dictionary
      fp      : value of fixed point
      mytol   : tolerance, default as 10^{-6}

    Returns :
      Whether it is a correct fixed point: True/False
    """

    drEdt, drIdt = utils.EIderivs(x_fp[0], x_fp[1], **pars)

    return np.sum(drEdt**2) < mytol

def check_fp_E(pars, x_fp, mytol=1e-3):
    """
    Verify (drE/dt)^2 + (drI/dt)^2< mytol

    Args:
      pars    : Parameter dictionary
      fp      : value of fixed point
      mytol   : tolerance, default as 10^{-6}

    Returns :
      Whether it is a correct fixed point: True/False
    """

    drEdt = utils.EIderivs_E(x_fp, **pars)

    return np.sum(drEdt**2) < mytol

def check_fp_E2(pars, x_fp, mytol=1e-6):
    """
    Verify (drE/dt)^2 + (drI/dt)^2< mytol

    Args:
      pars    : Parameter dictionary
      fp      : value of fixed point
      mytol   : tolerance, default as 10^{-6}

    Returns :
      Whether it is a correct fixed point: True/False
    """

    drEdt = utils.EIderivs_E2(x_fp, **pars)

    return np.sum(drEdt**2) < mytol