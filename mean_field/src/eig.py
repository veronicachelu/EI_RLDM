import numpy as np
from functools import partial

def get_drEdt_drE(fp, tau_E, a_E, b_E, d_E, df_E, dimA_E, dimA_I,
             wEE, wIE, I_ext_E, **other_pars):
    """
    Compute drEdt_drE

    Args:
      fp   : fixed point (E, I), array
      Other arguments are parameters of the Wilson-Cowan model

    Returns:
      J    : the 2x2 Jacobian matrix
    """
    rE, rI = fp

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])

    rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
    rI_wIE = np.einsum("j,jk->k", rI, W_IE) if dimA_I > 1 else rI * wIE

    dF_E = partial(df_E,  a=a_E, b=b_E, d=d_E)
    # Calculate the J[0,0]
    dFE = dF_E(rE_wEE - rI_wIE + I_ext_E)
    dFE_wEE = np.einsum("j,jk->k", dFE, wEE)  if dimA_E > 1 else dFE * wEE
    dGdrE = (-1 + dFE_wEE) / tau_E
    return dGdrE

def get_drE0dt_drE0(fp, rI, tau_E, a_E, b_E, d_E, df_E, dimA_E, dimA_I,
                    wEE, wIE, I_ext_E, **other_pars):
    """
    Compute drEdt_drE

    Args:
      fp   : fixed point (E, I), array
      Other arguments are parameters of the Wilson-Cowan model

    Returns:
      J    : the 2x2 Jacobian matrix
    """
    rE = fp

    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])

    rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
    rI_wIE = np.einsum("j,jk->k", rI, W_IE) if dimA_I > 1 else rI * wIE

    dF_E = partial(df_E,  a=a_E, b=b_E, d=d_E)
    # Calculate the J[0,0]
    dFE_0 = dF_E(rE_wEE - rI_wIE + I_ext_E)[0]
    drE0dt_drE0 = (-1 + (1+wEE) * dFE_0) / tau_E
    return drE0dt_drE0


def get_eig_Jacobian(fp,
                     tau_E, a_E, b_E, d_E, df_E, dimA_E,
                     wEE, wEI, I_ext_E,
                     tau_I, a_I, b_I, d_I, df_I, dimA_I,
                     wIE, wII, I_ext_I, **other_pars):
    """Compute eigenvalues of the Wilson-Cowan Jacobian matrix at fixed point."""
    # Initialization
    rE, rI = fp
    J = np.zeros((2, 2))
    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])

    rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
    rI_wIE = np.einsum("j,jk->k", rI, W_IE) if dimA_I > 1 else rI * wIE

    dF_E = partial(df_E,  a=a_E, b=b_E, d=d_E)
    # Compute the four elements of the Jacobian matrix
    dFE = dF_E(rE_wEE - rI_wIE + I_ext_E)
    dFE_wEE = np.einsum("j,jk->k", dFE, wEE) if dimA_E > 1 else dFE * wEE
    drEdt_drE = (-1 + dFE_wEE) / tau_E
    J[0, 0] = drEdt_drE

    dFE_wIE = np.einsum("j,jk->k", dFE, wIE) if dimA_I > 1 else dFE * wIE
    drEdt_drI = -dFE_wIE / tau_E
    J[0, 1] = drEdt_drI

    W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])
    W_II = np.array([[1 + wII, 1 - wII], [1 - wII, 1 + wII]])
    rE_wEI = np.einsum("ji,jk->ki", rE, W_EI) if dimA_E > 1 else rE * wEI
    rI_wII = np.einsum("ji,jk->ki", rI, W_II) if dimA_I > 1 else rI * wII

    dF_I = partial(df_I,  a=a_I, b=b_I, d=d_I)
    dFI = dF_I(rE_wEI - rI_wII + I_ext_I)
    dFI_wEI = np.einsum("j,jk->k", dFI, wEI) if dimA_E > 1 else dFI * wEI
    drIdt_drE = dFI_wEI / tau_I
    J[1, 0] = drIdt_drE

    dFI_wII = np.einsum("j,jk->k", dFI, wII) if dimA_I > 1 else dFI * wII
    drIdt_drI = (-1 - dFI_wII) / tau_I
    J[1, 1] = drIdt_drI

    # Compute and return the eigenvalues
    evals = np.linalg.eig(J)[0]
    return evals, J



def get_eig_Jacobian_E(fp, rI, tau_E, a_E, b_E, d_E, df_E, dimA_E,
                     wEE, I_ext_E, dimA_I,
                     wIE, **other_pars):
    """Compute eigenvalues of the Wilson-Cowan Jacobian matrix at fixed point."""
    # Initialization
    rE = fp
    J = np.zeros((2, 2))
    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])

    rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
    rI_wIE = np.einsum("j,jk->k", rI, W_IE) if dimA_I > 1 else rI * wIE

    dF_E = partial(df_E,  a=a_E, b=b_E, d=d_E)
    # Compute the four elements of the Jacobian matrix
    dFE_0 = dF_E(rE_wEE - rI_wIE + I_ext_E)[0]
    drE0dt_drE0 = (-1 + (1+wEE) * dFE_0) / tau_E
    # drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E)) / tau_E
    #drE0dt_drE0 = (-1 + (1+w_EE)dFE(rE_wEE - rI_wIE + I_ext_E)[0]) / tau_E
    J[0, 0] = drE0dt_drE0

    drE0dt_drE1 = (1-wEE) * dFE_0 / tau_E
    #drE0dt_drE1 = ((1-w_EE)dFE(rE_wEE - rI_wIE + I_ext_E)[0]) / tau_E
    J[0, 1] = drE0dt_drE1

    dFE_1 = dF_E(rE_wEE - rI_wIE + I_ext_E)[1]
    drE1dt_drE0 = (1-wEE) * dFE_1 / tau_E
    #drE1dt_drE0 = ((1-w_EE)dFE(rE_wEE - rI_wIE + I_ext_E)[1]) / tau_E
    J[1, 0] = drE1dt_drE0

    drE1dt_drE1 = (-1 + (1+wEE) * dFE_1) / tau_E
    #drE1dt_drE1 = (-1 + (1+w_EE)dFE(rE_wEE - rI_wIE + I_ext_E)[1]) / tau_E
    J[1, 1] = drE1dt_drE1

    # Compute and return the eigenvalues
    evals = np.linalg.eig(J)[0]
    return evals

def get_eig_Jacobian_E2(fp, tau_E, a_E, b_E, d_E, df_E, dimA_E,
                        a_I, b_I, d_I, f_I, rE_init, df_I,
                       wEE, wEI, I_ext_E, dimA_I, noise_mean,
                       wIE, **other_pars):
    """Compute eigenvalues of the Wilson-Cowan Jacobian matrix at fixed point."""
    # Initialization
    rE = fp
    J = np.zeros((2, 2))
    W_EE = np.array([[1 + wEE, 1 - wEE], [1 - wEE, 1 + wEE]])
    W_IE = np.array([[1 + wIE, 1 - wIE], [1 - wIE, 1 + wIE]])
    W_EI = np.array([[1 + wEI, 1 - wEI], [1 - wEI, 1 + wEI]])
    F_I = partial(f_I, a=a_I, b=b_I, d=d_I)
    rE_wEI = np.einsum("j,jk->k", rE, W_EI) if dimA_E > 1 else rE * wEI
    rI = F_I(rE_wEI)
    rE_wEE = np.einsum("j,jk->k", rE, W_EE) if dimA_E > 1 else rE * wEE
    rI_wIE = np.einsum("j,jk->k", rI, W_IE) if dimA_I > 1 else rI * wIE

    dF_E = partial(df_E,  a=a_E, b=b_E, d=d_E)
    dF_I = partial(df_I,  a=a_I, b=b_I, d=d_I)
    # Compute the four elements of the Jacobian matrix
    dFE_0 = dF_E(rE_wEE - rI_wIE + I_ext_E + noise_mean)[0]
    dFI_0 = dF_I(rE_wEI)[0]
    dFI_1 = dF_I(rE_wEI)[1]
    # rE = F_E(rE_wEE - rI_wIE + I_ext_E + noise_mean)
    drE0dt_drE0 = (-1 + ((1+wEE) - (1+wIE) * dFI_0 * (1+wEI) - (1-wIE) * dFI_1 * (1-wEI)) * dFE_0) / tau_E
    # drEdt = (-rE + F_E(rE_wEE - rI_wIE + I_ext_E + noise_mean)) / tau_E
    # drE0dt_drE0 = (-1 + [(1+w_EE) - drI_wIE_drE0]dFE(rE_wEE - rI_wIE + I_ext_E + noise_mean)[0]) / tau_E
    #             = (drE0_drE0 - 1) / tau_E
    J[0, 0] = drE0dt_drE0

    drE0dt_drE1 = ((1-wEE) - (1+wIE) * dFI_0 * (1-wEI) - (1-wIE) * dFI_1 * (1+wEI)) * dFE_0 / tau_E
    #drE0dt_drE1 = ([(1-w_EE) - drI_wIE_drE1]dFE(rE_wEE - rI_wIE + I_ext_E + noise_mean)[0]) / tau_E
    #            = drE0_drE1 / tau_E
    J[0, 1] = drE0dt_drE1

    dFE_1 = dF_E(rE_wEE - rI_wIE + I_ext_E + noise_mean)[1]
    drE1dt_drE0 = ((1-wEE) - (1-wIE) * dFI_0 * (1+wEI) - (1+wIE) * dFI_1 * (1-wEI)) * dFE_1 / tau_E
    #drE1dt_drE0 = ([(1-w_EE) - drI_wIE_drE0]dFE(rE_wEE - rI_wIE + I_ext_E + noise_mean)[1]) / tau_E
    #            = drE1_drE0 / tau_E
    J[1, 0] = drE1dt_drE0

    drE1dt_drE1 = (-1 + ((1+wEE) - (1-wIE) * dFI_0 * (1-wEI) - (1+wIE) * dFI_1 * (1+wEI)) * dFE_1) / tau_E
    #drE1dt_drE1 = (-1 + [(1+w_EE) - drI_wIE_drE1]dFE(rE_wEE - rI_wIE + I_ext_E + noise_mean)[1]) / tau_E
    #            =  (drE1_drE1 - 1) / tau_E
    J[1, 1] = drE1dt_drE1

    # drE0_drE1 = [(1-w_EE) - drI_wIE_drE1] dF_E(rE_wEE - rI_wIE + I_ext_E + noise_mean)[0]
    # drE1_drE0 = [(1-w_EE) - drI_wIE_drE0] dF_E(rE_wEE - rI_wIE + I_ext_E + noise_mean)[1]

    # (1 + wIE)(1 + wEI)* dFI_0 + (1 - wIE)*(1 - wEI) * dFI_1 = (1 + wEE)
    stability_cond = ((1 + wEE),
                      (1 + wIE)*(1 + wEI)* dFI_0 + (1 - wIE)*(1 - wEI) * dFI_1,
                      (1-wIE) * dFI_0 * (1-wEI) + (1+wIE) * dFI_1 * (1+wEI))
    # Compute and return the eigenvalues
    evals = np.linalg.eig(J)[0]
    return evals, J, stability_cond
