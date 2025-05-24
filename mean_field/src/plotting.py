# @title Plotting Functions

# Imports
import importlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from src import utils
utils = importlib.reload(utils)

# Constants
FONTSIZE = 11
FONTSIZE2 = 9
COLORS = plt.cm.get_cmap('Set1').colors
colors = [mpl.colors.rgb2hex(color[:3]) for color in COLORS]


def plot_FI_inverse(x, a, b, d, F_inv):
    f, ax = plt.subplots()
    ax.plot(x, F_inv(x, a=a, b=b, d=d))
    ax.set(xlabel="$x$", ylabel="$F^{-1}(x)$")


def plot_FI_EI(x, FI_exc, FI_inh, title, x_label, y_label):
    fig = plt.figure()
    fig.suptitle(title)
    plt.plot(x, FI_exc, colors[0], label=x_label)
    plt.plot(x, FI_inh, colors[1], label=y_label)
    plt.legend(loc='lower right')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.show()


def my_test_plot(range_t, stim_start, stim_end, rE=None, rI=None, ord=0):
    plot_size = len(rE) if rE is not None else len(rI) if rI is not None else 1
    plot_size2 = 2 if rE is not None and rI is not None else 1
    if ord == 1:
        plot_size2, plot_size = plot_size, plot_size2
    fig, axes = plt.subplots(plot_size2, plot_size, figsize=(3 * plot_size, 2 * plot_size2),
                      sharex='col', sharey='row',
                      squeeze=False,
                      # gridspec_kw={'hspace': 0.1}
                           )
    if rE is not None:
        for i, rE1 in enumerate(rE):
            ax1 = axes[0, i] if ord == 0 else axes[i, 0]
            ax1.axvspan(stim_start, stim_end, color="gray", alpha=0.3)
            mean_rE1 = np.mean(rE1, axis=0)
            std_rE1 = np.std(rE1, axis=0)
            if mean_rE1.shape[0] > 1:
                ax1.plot(range_t, mean_rE1[0], color=colors[0], ls="-", label='E1', alpha=0.7)
                ax1.fill_between(range_t, mean_rE1[0] - std_rE1[0], mean_rE1[0] + std_rE1[0],
                                 color=colors[0], alpha=0.3)
                ax1.plot(range_t, mean_rE1[1], colors[1], ls="-", label='E2', alpha=0.7)
                ax1.fill_between(range_t, mean_rE1[1] - std_rE1[1], mean_rE1[1] + std_rE1[1],
                                 color=colors[1], alpha=0.3)
            else:
                ax1.plot(range_t, rE1, colors[0], label='E')
            ax1.set_xlabel('t (ms)', fontsize=FONTSIZE)
            ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            ax1.tick_params(axis='both', which='minor', labelsize=FONTSIZE)
            ax1.legend(loc='best', fontsize=FONTSIZE)
        if ord == 0:
            axes[0,0].set_ylabel(r'$z_E$', fontsize=FONTSIZE)
        if ord == 1:
            for ax in axes.flat:
                ax.set_ylabel(r'$z_E$', fontsize=FONTSIZE)
    if rI is not None:
        for i, rI1 in enumerate(rI):
            ax1 = axes[1, i] if rE is not None else axes[0, i]
            ax1.axvspan(stim_start, stim_end, color="gray", alpha=0.3)
            mean_rI1 = np.mean(rI1, axis=0)
            std_rI1 = np.std(rI1, axis=0)
            if rI1.size > 1:
                ax1.plot(range_t, mean_rI1[0], color=colors[0], ls="--", label='I1', alpha=0.7)
                ax1.fill_between(range_t, mean_rI1[0] - std_rI1[0], mean_rI1[0] + std_rI1[0],
                                 color=colors[0], alpha=0.3)
                ax1.plot(range_t, mean_rI1[1], color=colors[1], ls="--", label='I2', alpha=0.7)
                ax1.fill_between(range_t, mean_rI1[1] - std_rI1[1], mean_rI1[1] + std_rI1[1],
                                 color=colors[1], alpha=0.3)
            else:
                ax1.plot(range_t, rI1, colors[1], label='I')
            ax1.set_xlabel('t (ms)')
            ax1.legend(loc='best')

        if rE is None:
            axes[0,0].set_ylabel(r'$z_I$')
        else:
            axes[1,0].set_ylabel(r'$z_I$')
    plt.tight_layout()
    plt.show()



def my_test_plot2(range_t, stim_start, stim_end, labels, rE=None, rI=None, ord=0):
    FONTSIZE=8
    plot_size = len(rE) if rE is not None else len(rI) if rI is not None else 1
    plot_size2 = 2 if rE is not None and rI is not None else 1
    if ord == 1:
        plot_size2, plot_size = plot_size, plot_size2
    fig, axes = plt.subplots(plot_size2, plot_size, figsize=(4 * 2/3 * plot_size, 1.5 * 2/3 * plot_size2),
                             sharex='col', sharey='all',
                             squeeze=False,
                             # gridspec_kw={'hspace': 0.1}
                             )
    if rE is not None:
        for i, rE1 in enumerate(rE):
            label = labels[i]
            ax1 = axes[0, i] if ord == 0 else axes[i, 0]
            ax1.text(-1.5, 3. + 0. * i, label, fontsize=FONTSIZE,  color=colors[2+i])
            ax1.axvspan(stim_start, stim_end, color="gray", alpha=0.3)
            mean_rE1 = np.mean(rE1, axis=0)
            std_rE1 = np.std(rE1, axis=0)
            if mean_rE1.shape[0] > 1:
                ax1.plot(range_t, mean_rE1[0], color=colors[0], ls="-", label='E1', alpha=0.7)
                ax1.fill_between(range_t, mean_rE1[0] - std_rE1[0], mean_rE1[0] + std_rE1[0],
                                 color=colors[0], alpha=0.3)
                ax1.plot(range_t, mean_rE1[1], colors[1], ls="-", label='E2', alpha=0.7)
                ax1.fill_between(range_t, mean_rE1[1] - std_rE1[1], mean_rE1[1] + std_rE1[1],
                                 color=colors[1], alpha=0.3)
            else:
                ax1.plot(range_t, rE1, colors[0], label='E')
            # ax1.set_title(label, color=colors[2+i], fontsize=FONTSIZE)
            # ax1.set_xlabel('t (ms)', fontsize=FONTSIZE)
            ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            ax1.tick_params(axis='both', which='minor', labelsize=FONTSIZE)
            # ax1.legend(loc='best', fontsize=FONTSIZE)
        if ord == 0:
            axes[0,0].set_ylabel(r'$r_E$', fontsize=FONTSIZE)
        if ord == 1:
            for ax in axes.flat:
                ax.set_ylabel(r'$r_E$', fontsize=FONTSIZE)
    if rI is not None:
        for i, rI1 in enumerate(rI):
            ax1 = axes[1, i] if rE is not None else axes[0, i]
            ax1.axvspan(stim_start, stim_end, color="gray", alpha=0.3)
            mean_rI1 = np.mean(rI1, axis=0)
            std_rI1 = np.std(rI1, axis=0)
            if rI1.size > 1:
                ax1.plot(range_t, mean_rI1[0], color=colors[0], ls="--", label='I1', alpha=0.7)
                ax1.fill_between(range_t, mean_rI1[0] - std_rI1[0], mean_rI1[0] + std_rI1[0],
                                 color=colors[0], alpha=0.3)
                ax1.plot(range_t, mean_rI1[1], color=colors[1], ls="--", label='I2', alpha=0.7)
                ax1.fill_between(range_t, mean_rI1[1] - std_rI1[1], mean_rI1[1] + std_rI1[1],
                                 color=colors[1], alpha=0.3)
            else:
                ax1.plot(range_t, rI1, colors[1], label='I')

            ax1.legend(loc='best')

        if rE is None:
            axes[0,0].set_ylabel(r'$r_I$')
        else:
            axes[1,0].set_ylabel(r'$r_I$')

    axes[2,0].set_xlabel('t (ms)', fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()


def plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI,
                    label1='E nullcline', label2='I nullcline',
                    x_label=r'$r_E$', y_label=r'$r_I$'):

    plt.figure()
    plt.plot(Exc_null_rE, Exc_null_rI, colors[0], label=label1)
    plt.plot(Inh_null_rE, Inh_null_rI, colors[1], label=label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.show()


def my_plot_nullcline(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI,
                      label1='E nullcline', label2='I nullcline',
                      x_label=r'$r_E$', y_label=r'$r_I$', ls="-"):
    plt.plot(Exc_null_rE, Exc_null_rI, colors[0], ls=ls, label=label1)
    plt.plot(Inh_null_rE, Inh_null_rI, colors[1], ls=ls, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')


def my_plot_vector(pars, my_n_skip=2, myscale=5):
    EI_grid = np.linspace(0., 1., 20)
    rE, rI = np.meshgrid(EI_grid, EI_grid)
    drEdt, drIdt = utils.EIderivs(rE, rI, **pars)

    n_skip = my_n_skip

    plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
               drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
               angles='xy', scale_units='xy', scale=myscale, facecolor='c')

    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')


def my_plot_trajectory(pars, mycolor, x_init, mylabel):
    pars = pars.copy()
    pars['rE_init'], pars['rI_init'] = x_init[0], x_init[1]
    rE_tj, rI_tj = simulate_wc(**pars)

    plt.plot(rE_tj, rI_tj, color=mycolor, label=mylabel)
    plt.plot(x_init[0], x_init[1], 'o', color=mycolor, ms=8)
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')


def my_plot_trajectories(pars, dx, n, mylabel):
    """
    Solve for I along the E_grid from dE/dt = 0.

    Expects:
    pars    : Parameter dictionary
    dx      : increment of initial values
    n       : n*n trjectories
    mylabel : label for legend

    Returns:
      figure of trajectory
    """
    pars = pars.copy()

    for ie in range(n):
        for ii in range(n):
            pars['rE_init'], pars['rI_init'] = [dx * ie], [dx * ii]
            rE_tj, rI_tj = utils.simulate_wc(**pars)
            if (ie == n-1) & (ii == n-1):
                plt.plot(rE_tj, rI_tj, 'gray', alpha=0.8, label=mylabel)
            else:
                plt.plot(rE_tj, rI_tj, 'gray', alpha=0.8)

    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')


def plot_complete_analysis(pars):
    plt.figure(figsize=(7.7, 6.))

    # plot example trajectories
    my_plot_trajectories(pars, 0.2, 6,
                         'Sample trajectories \nfor different init. conditions')
    my_plot_trajectory(pars, 'orange', [0.6, 0.8],
                       'Sample trajectory for \nlow activity')
    my_plot_trajectory(pars, 'm', [0.6, 0.6],
                       'Sample trajectory for \nhigh activity')

    # plot nullclines
    my_plot_nullcline(pars)

    # plot vector field
    EI_grid = np.linspace(0., 1., 20)
    rE, rI = np.meshgrid(EI_grid, EI_grid)
    drEdt, drIdt = EIderivs(rE, rI, **pars)
    n_skip = 2
    plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
               drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
               angles='xy', scale_units='xy', scale=5., facecolor='c')

    plt.legend(loc=[1.02, 0.57], handlelength=1)
    plt.show()


def plot_fp(x_fp, ax=None, eig=None, J=None, stability_cond=None, position=(1.5, 1.1), color='k', marker='o', rotation=0):
    ax.plot(x_fp[0], x_fp[1], color=color, marker=marker, ms=6, alpha=0.7)
    if eig is not None:
        # zE = r"$z_E$"
        lam = r"$\lambda$"
        eigenvalues = []
        # J = np.array(J, dtype=float)  # Ensure J contains numeric values
        # formatted_J = f'[{J[0, 0]:.1f}, {J[0, 1]:.1f}\n {J[1, 0]:.1f}, {J[1, 1]:.1f}]'
        # J00 = f'{J[0, 0]:.1f}'
        # J01 = f'{J[0, 1]:.1f}'
        # J10 = f'{J[1, 0]:.1f}'
        # J11 = f'{J[1, 1]:.1f}'


        # formatted_J = (r'$\begin{{bmatrix}} $'+
        #                J00 + r' & ' + J01 + r'$ \\ $'
        #               + J10 + r' & ' + J11 + r'$\end{{bmatrix}}$')
        for i in range(2):
            if np.iscomplex(eig[i]):
                real_part = eig[i].real
                imag_part = eig[i].imag
                eigenvalues.append(f'{real_part:.1f}+{imag_part:.1f}i')
            else:
                real_part = eig[i]
                eigenvalues.append(f'{real_part:.1f}')
        # ax.text(x_fp[0] + position[0], x_fp[1] + position[1],
        #          # f'{zE}:{x_fp[0]:.1f}, {x_fp[1]:.1f}'
        #         # formatted_J,
        #         # rf'$\lambda = ({eigenvalues[0]:.1f}, {eigenvalues[1]:.1f})$',
        #         # f'J:{formatted_J}\n'
        #         # f'({stability_cond[0]:.1f}, {stability_cond[1]:.1f},{stability_cond[2]:.1f})\n'
        #          f'{lam}:({eigenvalues[0]}, {eigenvalues[1]})',
        #          horizontalalignment='center', fontsize=FONTSIZE2, verticalalignment='bottom',
        #          rotation=rotation)
    # else:
    #     ax.text(x_fp[0] + position[0], x_fp[1] + position[1],
    #              f'({x_fp[0]:.3f}, {x_fp[1]:.3f})', fontsize=FONTSIZE2,
    #              horizontalalignment='center', verticalalignment='bottom',
    #              rotation=rotation)



def plot_nullcline_diffwEE(pars, Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI):
    """
      plot nullclines for different values of wEE
    """

    plt.figure(figsize=(12, 5.5))
    plt.subplot(121)
    plt.plot(Exc_null_rE, Exc_null_rI, 'b', label='E nullcline')
    plt.plot(Inh_null_rE, Inh_null_rI, 'r', label='I nullcline')
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')
    plt.legend(loc='best')

    plt.subplot(222)
    pars['rE_init'], pars['rI_init'] = [0.2], [0.2]
    rE, rI = utils.simulate_wc(**pars)
    plt.plot(pars['range_t'], rE[0, 0], colors[0], label='E population', clip_on=False)
    plt.plot(pars['range_t'], rI[0, 0], colors[1], label='I population', clip_on=False)
    plt.ylabel('Activity')
    plt.legend(loc='best')
    plt.ylim(-0.05, 1.05)
    plt.title('E/I activity\nfor different initial conditions',
              fontweight='bold')

    plt.subplot(224)
    pars['rE_init'], pars['rI_init'] = [0.4], [0.1]
    rE, rI = utils.simulate_wc(**pars)
    plt.plot(pars['range_t'], rE[0, 0], colors[0], label='E population', clip_on=False)
    plt.plot(pars['range_t'], rI[0, 0], colors[1], label='I population', clip_on=False)
    plt.xlabel('t (ms)')
    plt.ylabel('Activity')
    plt.legend(loc='best')
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()