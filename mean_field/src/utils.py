# @title Helper Functions
# Imports
import importlib
import numpy as np
from functools import partial
from numpy.linalg import inv

from src import tuning  # Built-in module
tuning = importlib.reload(tuning)

def thesis_pars(**kwargs):
    pars = {}

    # simulation parameters
    pars['time'] = 50.  # Total duration of simulation [ms]
    pars['dt'] = 1  # Simulation time step [ms]
    # Vector of discretized time points [ms]
    pars['range_t'] = np.arange(0, pars['time'], pars['dt'])
    pars['T'] = pars['range_t'].size
    pars['n_seeds'] = 10  # Number of trials
    pars['rE_init'] = [0., 0.]  # Initial value of E
    pars['rI_init'] = [0., 0.]  # Initial value of I
    pars['rNoise_init'] = [0., 0.]  # Initial value of I

    # Excitatory parameters
    pars['tau_E'] = 1  # Timescale of the E population [ms]
    pars['a_E'] = 1 # Gain of the E population
    pars['b_E'] = 0 # Threshold of the E population
    pars['d_E'] = 1  # Slope of the E population
    pars['f_E'] = tuning.F_elu  # tuning fn of the E population
    pars['f_inv_E'] = tuning.F_elu_inv # inverse of the tuning fn of the E population
    pars['df_E'] = tuning.dF_elu
    # pars['f_E'] = tuning.F_sigmoid  # tuning fn of the E population
    # pars['f_inv_E'] = tuning.F_sigmoid_inv # inverse of the tuning fn of the E population
    # pars['df_E'] = tuning.dF_sigmoid
    pars['dimA_E'] = 2  # action dim

    # Inhibitory parameters
    pars['tau_I'] = 1  # Timescale of the I population [ms]
    pars['a_I'] = 1.  # Gain of the I population
    pars['b_I'] = 0  # Threshold of the I population
    pars['d_I'] = 1  # Slope of the I population
    # pars['f_inv_I'] = tuning.F_sigmoid_inv # inverse of the tuning fn of the I population
    # pars['f_I'] = tuning.F_sigmoid  # tuning fn of the I population
    # pars['df_I'] = tuning.dF_sigmoid
    pars['f_I'] = tuning.F_linear  # tuning fn of the I population
    pars['f_inv_I'] = tuning.F_linear_inv # inverse of the tuning fn of the I population
    pars['df_I'] = tuning.dF_linear
    # pars['f_I'] = tuning.F_elu  # tuning fn of the E population
    # pars['f_inv_I'] = tuning.F_elu_inv # inverse of the tuning fn of the E population
    # pars['df_I'] = tuning.dF_elu
    pars['dimA_I'] = 2  # action dim

    # Noise parameters
    pars['tau_noise'] = 1  # Timescale of the E population [ms]
    pars["noise_mean"] = 0  # Mean of input noise
    pars["noise_std"] = 0.02  # Standard deviation of input noise

    # Stimulation parameters
    pars["stimulus_strength"] = 1  # Range of stimulus strengths
    pars["coherence_range"] = (0.5, 0.5)  # Bias in stimulus input
    pars["stimulation_period"] = (20., 30.)  # Fraction of time for the stimulation period

    # Connection strength
    pars['wEE'] = 0.  # E to E
    pars['wEI'] = 0.  # I to E
    pars['wIE'] = 0.  # E to I
    pars['wII'] = 0.  # I to I

    pars['I_ext_E'] = 0
    pars['I_ext_I'] = 0

    # External parameters if any
    for k in kwargs:
        pars[k] = kwargs[k]

    return pars


def get_inputs(n_seeds, dimA_E, dimA_I, T, dt, stimulus_strength, coherences,
               stim_start=0, stim_end=None):
    stim_end = T-1 if stim_end is None else stim_end
    I_ext_E = np.zeros((n_seeds, dimA_E, T))
    I_ext_I = np.zeros((n_seeds, dimA_I, T))
    for t in range(0, T - 1):
        time = t * dt
        if stim_start <= time <= stim_end:
            I_ext_E[:, 0, t] = stimulus_strength * (1 - coherences)  # Population 1
            if dimA_E > 1:
                I_ext_E[:, 1, t] = stimulus_strength * (1 + coherences)  # Population 2
    return I_ext_E, I_ext_I

def get_noise(rngs, n_seeds, dimA, T, noise_mean, noise_std):
    noise = np.zeros((n_seeds, dimA, T))
    for t in range(T - 1):
        noise[:, :, t] = np.array([rng.normal(noise_mean, noise_std, size=dimA)
                          for rng in rngs])

    return noise

def simulate_wc(T, n_seeds, dt,
                tau_E, a_E, b_E, d_E, f_E, dimA_E,
                tau_I, a_I, b_I, d_I, f_I, dimA_I,
                noise, tau_noise,
                wEE, wEI, wIE, wII,
                I_ext_E, I_ext_I,
                rE_init, rI_init, rNoise_init,
                **other_pars):
    """
    Simulate the Wilson-Cowan equations

    Args:
      Parameters of the Wilson-Cowan model

    Returns:
      rE, rI (arrays) : Activity of excitatory and inhibitory populations
    """
    # Initialize activity arrays
    rE = np.zeros((n_seeds, dimA_E, T))  # Excitatory populations (E1, E2)
    rI = np.zeros((n_seeds, dimA_I, T))  # Inhibitory populations (I1, I2)
    rNoise = np.zeros((n_seeds, dimA_E, T))

    F_E = partial(f_E, d=d_E, a=a_E, b=b_E)
    F_I = partial(f_I, d=d_I, a=a_I, b=b_I)

    # Connectivity weight matrix for excitatory and inhibitory populations
    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])
    W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])
    # W_II = np.array([[1 + wII, 1 - wII], [1 - wII, 1 + wII]])

    # Set initial conditions for all seeds
    if dimA_E > 1:
        rE[:, 0, 0] = np.array(rE_init[0])
        rE[:, 1, 0] = np.array(rE_init[1])
    else:
        rE[:, :, 0] = np.array(rE_init)
    if dimA_I > 1:
        rI[:, 0, 0] = np.array(rI_init[0])
        rI[:, 1, 0] = np.array(rI_init[1])
    else:
        rI[:, :, 0] = np.array(rI_init)
    rNoise[:, :, 0] = rNoise_init

    # Simulate the Wilson-Cowan equations
    for t in range(T - 1):
        dNoise = (-rNoise[:, :, t] + noise[:, :, t]) / tau_noise

        rE_wEE = np.einsum("ij,jk->ik", rE[:, :, t], W_EE) if dimA_E > 1 else rE[:, :, t] * wEE
        rE_wEI = np.einsum("ij,jk->ik", rE[:, :, t], W_EI) if dimA_E > 1 else rE[:, :, t] * wEI
        rI_wIE = np.einsum("ij,jk->ik", rI[:, :, t], W_IE) if dimA_I > 1 else rI[:, :, t] * wIE
        # rI_wII = np.einsum("ij,jk->ik", rI[:, :, t], W_II) if dimA_I > 1 else rI[:, :, t] * wII

        # Calculate the derivative of the E population
        drE = (-rE[:, :, t] + F_E(rE_wEE - rI_wIE + I_ext_E[:, :, t] + rNoise[:, :, t])) / tau_E

        # Calculate the derivative of the I population
        drI = (-rI[:, :, t] + F_I(rE_wEI)) / tau_I

        # Update using Euler's method
        rE[:, :, t + 1] = rE[:, :, t] + drE * dt
        rI[:, :, t + 1] = rI[:, :, t] + drI * dt
        rNoise[:, :, t + 1] = rNoise[:, :, t] + dNoise * dt

    return rE, rI


def get_E_nullcline(rE, a_E, b_E, d_E, wEE, wIE, I_ext_E, f_inv_E, dimA_E, dimA_I, **other_pars):
    """
    Solve for rI along the rE from drE/dt = 0.

    Args:
      rE    : response of excitatory population
      a_E, b_E, wEE, wEI, I_ext_E : Wilson-Cowan excitatory parameters
      Other parameters are ignored

    Returns:
      rI    : values of inhibitory population along the nullcline on the rE
    """

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])
    # rI * wIE = (rE * wEE - F_inv_E(rE[:, :, t]) + I_ext_E[:, :, t] + rNoise[:, :, t])
    rE_wEE = np.einsum("ji,jk->ki", rE, W_EE) if dimA_E > 1 else rE * wEE

    inv_wIE = inv(W_IE) if dimA_I > 1 else 1 / wIE
    # calculate rI for E nullclines on rI
    F_inv_E = partial(f_inv_E, a=a_E, b=b_E, d=d_E)

    rI = inv_wIE * (rE_wEE - F_inv_E(rE) + I_ext_E)

    return rI

def get_II_nullcline(I_no, rE, rI, a_I, b_I, d_I, wEI, wII, I_ext_I, f_inv_I, dimA_E, dimA_I, **other_pars):
    """
    Solve for rI[2/1] along the rI[1/2], rE from drI[1/2]/dt = 0.

    Args:
      rI    : response of excitatory population
      rE    : response of inhibitory population
      a_I, b_I, d_I, wEI, wII, I_ext_I : Wilson-Cowan excitatory parameters
      Other parameters are ignored

    Returns:
      rI[2/1]    : values of inhibitory population along the nullcline on the rI1
    """

    W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])
    # 0 = drIdt = (-rI + F_I(rE_wEI - rI_wII + I_ext_I)) / tau_I
    # rI = F_I(rE_wEI - rI_wII + I_ext_I)
    # F_inv_I(rI) = rE_wEI - rI_wII + I_ext_I
    # 0 = rE_wEI - F_inv_I(rI) - rI_wII + I_ext_I
    # 0 = rE * wEI - F_inv_I(rI) - rI * wII + I_ext_I
    # 0 = (rE * wEI)[0] - F_inv_I(rI1) - rI1 * (1 + wII) - rI2 * (1 - wII) + I_ext_I1
    # rI2 * (1 - wII) = (rE * wEI)[0] - F_inv_I(rI1) - rI1 * (1 + wII) + I_ext_I1
    # rI2 = ((rE * wEI)[0] - F_inv_I(rI1) - rI1 * (1 + wII) + I_ext_I1)/(1 - wII)
    rE_wEI = np.einsum("ji,jk->ki", rE, W_EI) if dimA_E > 1 else rE * wEI
    F_inv_I = partial(f_inv_I, a=a_I, b=b_I, d=d_I)

    rI_opposite = (rE_wEI[I_no] - F_inv_I(rI[I_no]) - rI[I_no] * (1 + wII) + rE_wEI[I_no] + I_ext_I[I_no]) / (1 - wII)

    return rI_opposite

def get_EE_nullcline(E_no, rE, rI, a_E, b_E, d_E, wEE, wIE, I_ext_E, f_inv_E, dimA_E, dimA_I, **other_pars):
    """
    Solve for rE[2/1] along the rE[1/2], rI from drE[1/2]/dt = 0.

    Args:
      rE    : response of excitatory population
      rI    : response of inhibitory population
      a_E, b_E, d_E, wEE, wEI, I_ext_E, I_ext_I : Wilson-Cowan excitatory parameters
      Other parameters are ignored

    Returns:
      rE[2/1]    : values of excitatory population along the nullcline on the rE1
    """

    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])
    # 0 = drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E)) / tau_E
    # rE = F_E(rE_wEE - rI_wIE + I_ext_E)
    # F_inv_E(rE) = rE_wEE - rI_wIE + I_ext_E
    # 0 = rE_wEE - F_inv_E(rE) - rI_wIE + I_ext_E
    # 0 = rE * wEE - F_inv_E(rE) - rI * wIE + I_ext_E
    # 0 = rE1 * (1 + wEE) + rE2 * (1 - wEE) - F_inv_E(rE_1) - (rI * wIE)[0] + I_ext_E1
    # -rE2 * (1 - wEE) = rE1 * (1 + wEE) - (rI * wIE)[0] - F_inv_E(rE_1) + I_ext_E1
    # rE2 = (-rE1 * (1 + wEE) + (rI * wIE)[0] + F_inv_E(rE_1) - I_ext_E1)/(1 - wEE)

    rI_wIE = np.einsum("ji,jk->ki", rI, W_IE) if dimA_I > 1 else rI * wIE

    # calculate rI for E nullclines on rI
    F_inv_E = partial(f_inv_E, a=a_E, b=b_E, d=d_E)
    numerator = (-rE[E_no] * (1 + wEE) + rI_wIE[E_no] + F_inv_E(rE[E_no]) - I_ext_E[E_no])
    denominator = (1 - wEE)
    rE_opposite = np.divide(numerator, denominator, out=np.full_like(numerator, np.inf), where=denominator != 0)

    return rE_opposite

def get_Eop_from_E_nullcline(no, rE, a_E, b_E, d_E, a_I, b_I, wEE, wIE, wEI, I_ext_E, f_inv_E, noise_mean, **other_pars):
    """
    Compute z_{E1} from the given z_{E2} (rE2) according to the 'inverse' nullcline equation:

    z_{E1} = 1 / [ a_I(1 - w_IE)(1 + w_EI)
                   + a_I(1 + w_IE)(1 - w_EI)
                   - (1 - w_EE) ]
               * [ I_ext_E + 2*b_I
                   - f_inv_E(rE2, a_E, b_E, d_E, **other_pars)
                   - ( a_I(1 - w_IE)(1 - w_EI)
                       + a_I(1 + w_IE)(1 + w_EI)
                       - (1 + w_EE)
                     ) * rE2
                 ]

    Parameters
    ----------
    rE1 : float
        The value of z_{E2}.
    a_E, b_E, d_E : float
        Parameters for the excitatory population (used by f_inv_E).
    a_I, b_I, d_I : float
        Parameters for the inhibitory population.
    wEE, wIE, wEI, wII : float
        Synaptic weight parameters (E->E, I->E, E->I, I->I).
    I_ext_E : float
        External excitatory input (analogous to x2 in the math expression).
    f_inv_E : callable
        Inverse of the firing-rate function f_E. Must accept
        (value, a_E, b_E, d_E, **other_pars) as arguments.
    dimA_E, dimA_I : int
        Possible dimensional parameters (not used explicitly here, but passed
        for compatibility).
    other_pars : dict
        Dictionary for any additional parameters required by f_inv_E.

    Returns
    -------
    zE1 : float
        Computed value of z_{E1}.
    """

    # Denominator
    denom = (
            a_I * (1 - wIE) * (1 + wEI)
            + a_I * (1 + wIE) * (1 - wEI)
            - (1 - wEE)
    )

    # Numerator
    num = (
            I_ext_E[no] + noise_mean
            + 2 * b_I
            - f_inv_E(rE[no], a_E, b_E, d_E)
            - (
                    a_I * (1 - wIE) * (1 - wEI)
                    + a_I * (1 + wIE) * (1 + wEI)
                    - (1 + wEE)
            ) * rE[no]
    )

    rE_op = np.divide(num, denom, out=np.full_like(num, np.inf), where=denom != 0)

    return rE_op

def get_E1_from_E0_nullcline(rE0, a_E, b_E, d_E, a_I, b_I, d_I, wEE, wIE, wEI, wII, I_ext_E, f_inv_E, dimA_E, dimA_I, **other_pars):
    """
       Compute z_{E2} from the given z_{E1} (rE1) according to the nullcline equation:

       z_{E2} = 1 / [ a_I(1+w_IE)(1-w_EI) + a_I(1-w_IE)(1+w_EI) - (1 - w_EE) ]
                  * [ I_ext_E + 2*b_I
                      - f_inv_E(rE1, a_E, b_E, d_E, **other_pars)
                      - ( a_I(1+w_IE)(1+w_EI)
                          + a_I(1-w_IE)(1-w_EI)
                          + (1 + w_EE)
                        ) * rE1
                    ]

       Parameters
       ----------
       rE0 : float
           The value of z_{E1}.
       a_E, b_E, d_E : float
           Parameters for the excitatory population (used by f_inv_E).
       a_I, b_I, d_I : float
           Parameters for the inhibitory population.
       wEE, wIE, wEI, wII : float
           Synaptic weight parameters (E->E, I->E, E->I, I->I).
       I_ext_E : float
           External excitatory input (denoted x_1 in the math expression).
       f_inv_E : callable
           Inverse of the firing-rate function f_E. Must accept
           (value, a_E, b_E, d_E, **other_pars) as arguments.
       dimA_E, dimA_I : int
           Possible dimensional parameters (not used explicitly here, but passed
           for compatibility).
       other_pars : dict
           Dictionary for any additional parameters required by f_inv_E.

       Returns
       -------
       zE2 : float
           Computed value of z_{E2}.
       """
    # Denominator
    denom = (
             a_I * (1 - wIE) * (1 + wEI)
             + a_I * (1 + wIE) * (1 - wEI)
            - (1 - wEE)
    )

    # Numerator
    num = (
            I_ext_E[0]
            + 2 * b_I
            - f_inv_E(rE0, a_E, b_E, d_E)
            - (
                    a_I * (1 - wIE) * (1 - wEI)
                    + a_I * (1 + wIE) * (1 + wEI)
                    + (1 + wEE)
            ) * rE0
    )

    rE1 = np.divide(num, denom, out=np.full_like(num, np.inf), where=denom != 0)

    return rE1

def get_I_nullcline(rI, a_I, b_I, d_I, wEI, wII, I_ext_I, f_inv_I, dimA_E, dimA_I, **other_pars):
    """
    Solve for E along the rI from dI/dt = 0.

    Args:
      rI    : response of inhibitory population
      a_I, b_I, wIE, wII, I_ext_I : Wilson-Cowan inhibitory parameters
      Other parameters are ignored

    Returns:
      rE    : values of the excitatory population along the nullcline on the rI
    """
    W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])
    W_II = np.array([[1 + wII, 1 - wII], [1 - wII, 1 + wII]])
    rI_wII = np.einsum("ji,jk->ki", rI, W_II) if dimA_I > 1 else rI * wII
    # rE * wEI = F_inv_I(rI) + rI * wII - I_ext_I
    inv_wEI = inv(W_EI) if dimA_E > 1 else 1 / wEI
    # calculate rE for I nullclines on rI
    F_inv_I = partial(f_inv_I, a=a_I, b=b_I, d=d_I)
    rE = inv_wEI * (rI_wII + F_inv_I(rI) - I_ext_I)

    return rE


def EIderivs(rE, rI,
             tau_E, a_E, b_E, d_E, f_E, wEE, wEI, I_ext_E,
             tau_I, a_I, b_I, d_I, wIE, wII, f_I, I_ext_I,
             dimA_E, dimA_I,
             **other_pars):
    """Time derivatives for E/I variables (dE/dt, dI/dt)."""

    F_E = partial(f_E, a=a_E, b=b_E, d=d_E)
    F_I = partial(f_I, a=a_I, b=b_I, d=d_I)

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])
    W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])
    W_II = np.array([[1 + wII, 1 - wII], [1 - wII, 1 + wII]])

    rE_wEE = np.einsum("ji,jk->ki", rE, W_EE) if dimA_E > 1 else rE * wEE
    rE_wEI = np.einsum("ji,jk->ki", rE, W_EI) if dimA_E > 1 else rE * wEI
    rI_wIE = np.einsum("ji,jk->ki", rI, W_IE) if dimA_I > 1 else rI * wIE
    rI_wII = np.einsum("ji,jk->ki", rI, W_II) if dimA_I > 1 else rI * wII

    # Compute the derivative of rE
    drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E)) / tau_E

    # Compute the derivative of rI
    drIdt = (-rI + F_I(rE_wEI - rI_wII + I_ext_I)) / tau_I

    return drEdt, drIdt



def EIderivs_E(rE,
             tau_E, a_E, b_E, d_E, f_E, wEE, I_ext_E,
              wIE, rI_init,
             dimA_E, dimA_I,
             **other_pars):
    """Time derivatives for E/I variables (dE/dt, dI/dt)."""

    F_E = partial(f_E, a=a_E, b=b_E, d=d_E)

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])

    rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
    rI_wIE = np.einsum("j,jk->k", rI_init, W_IE) if dimA_I > 1 else rI_init * wIE

    # Compute the derivative of rE
    drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E)) / tau_E

    return drEdt

def EIderivs_E2(rE,
               tau_E, a_E, b_E, d_E, f_E, wEE, I_ext_E,
               wIE, a_I, b_I, d_I, f_I,wEI,
               dimA_E, dimA_I, noise_mean,
               **other_pars):
    """Time derivatives for E/I variables (dE/dt, dI/dt)."""

    F_E = partial(f_E, a=a_E, b=b_E, d=d_E)

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])
    W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])
    F_I = partial(f_I, a=a_I, b=b_I, d=d_I)
    rE_wEI = np.einsum("j,jk->k", rE, W_EI) if dimA_E > 1 else rE * wEI
    rI = F_I(rE_wEI)

    rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
    rI_wIE = np.einsum("j,jk->k", rI, W_IE) if dimA_I > 1 else rI * wIE

    # Compute the derivative of rE
    drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E + noise_mean)) / tau_E

    return drEdt


def batch_EIderivs_E(rE,
               tau_E, a_E, b_E, d_E, f_E, wEE, I_ext_E,
               wIE, rI_init,
               dimA_E, dimA_I,
               **other_pars):
    """Time derivatives for E/I variables (dE/dt, dI/dt)."""

    F_E = partial(f_E, a=a_E, b=b_E, d=d_E)

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])

    rE_wEE = np.einsum("ji,jk->ki", rE, W_EE) if dimA_E > 1 else rE * wEE
    rI_wIE = np.einsum("ji,jk->ki", rI_init, W_IE) if dimA_I > 1 else rI_init * wIE

    # Compute the derivative of rE
    drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E)) / tau_E

    return drEdt

def batch_EIderivs_E2(rE,
                     tau_E, a_E, b_E, d_E, f_E,
                      a_I, b_I, d_I, f_I,
                      wEE, I_ext_E, wEI,
                     wIE, noise_mean,
                     dimA_E, dimA_I,
                     **other_pars):
    """Time derivatives for E/I variables (dE/dt, dI/dt)."""

    F_E = partial(f_E, a=a_E, b=b_E, d=d_E)
    F_I = partial(f_I, a=a_I, b=b_I, d=d_I)

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])
    W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])

    rE_wEI = np.einsum("ji,jk->ki", rE, W_EI) if dimA_E > 1 else rE * wEI
    rE_wEE = np.einsum("ji,jk->ki", rE, W_EE) if dimA_E > 1 else rE * wEE
    rI = F_I(rE_wEI)
    rI_wIE = np.einsum("ji,jk->ki", rI, W_IE) if dimA_I > 1 else rI * wIE

    # Compute the derivative of rE
    drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E + noise_mean)) #/ tau_E

    return drEdt

