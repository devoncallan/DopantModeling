import matplotlib.pyplot as plt
from NRSS.checkH5 import checkH5
import numpy as np

from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

Q_LABEL = '$\it{q}$ (nm$^{-1}$)'
E_LABEL = 'Energy (eV)'
I_LABEL = 'Intensity (a.u.)'

Q_ROUNDING = 0.01
E_ROUNDING = 1
I_ROUNDING = 1

NUM_TICKS = 5

ANISOTROPY_CMAP = 'RdBu_r'
INTENSITY_CMAP = 'viridis'

Q_RANGE_DEFAULT = (0.1, 0.9)
E_RANGE_DEFAULT = (280, 290)
I_RANGE_DEFAULT = (1e-1, 1e7)

def add_title(fig, current_edge:str):
    fig.suptitle()

def add_text(ax, text:str):
    ax.text(0.975, 0.91, f'{text}', color='white', fontsize=15, transform=ax.transAxes, horizontalalignment='right')

def add_cbar(fig, im, label:str=''):

    cbar_ax = fig.add_axes([0.975, 0.145, 0.03, 0.715])
    cbar = fig.colorbar(im, cax=cbar_ax)

    plt.setp(cbar.ax.get_yticklabels(), ha='right')
    cbar.ax.tick_params(pad=37.5)
    cbar.ax.set_ylabel(label)

def generate_linear_ticks(tick_range, num_ticks, rounding_order):
    tick_min, tick_max = tick_range
    tick_spacing = (tick_max - tick_min) / (num_ticks - 1)
    tick_vals = [tick_min + i * tick_spacing for i in range(num_ticks)]
    rounded_tick_vals = [round(val / rounding_order) * rounding_order for val in tick_vals]

    # Determine the number of decimal places for formatting
    if type(rounding_order) == int:
        decimal_places = 0
    else:
        decimal_places = len(str(rounding_order).split('.')[1])
    format_specifier = f'{{:.{decimal_places}f}}'

    rounded_tick_strs = [format_specifier.format(value) for value in rounded_tick_vals]
    rounded_tick_strs[0] = ''
    rounded_tick_strs[-1] = ''
    
    return rounded_tick_vals, rounded_tick_strs

def generate_log_ticks(tick_range, num_ticks):
    tick_min, tick_max = tick_range
    log_min, log_max = np.log10(tick_min), np.log10(tick_max)

    # Generate logarithmically spaced values
    log_tick_vals = np.logspace(log_min, log_max, num=num_ticks)

    # Formatting the tick values for string representation in scientific notation
    rounded_tick_strs = ['' if i in [0, num_ticks - 1] else f'$10^{{{int(np.log10(val))}}}$' for i, val in enumerate(log_tick_vals)]

    return log_tick_vals, rounded_tick_strs

def format_q_xaxis(ax, q_range):

    q_min, q_max = q_range
    ax.set_xlabel(Q_LABEL)
    x_ticks, x_tick_strs = generate_linear_ticks(q_range, NUM_TICKS, Q_ROUNDING)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_strs)
    ax.set_xlim([q_min, q_max])

def format_E_yaxis(ax, E_range):

    E_min, E_max = E_range
    ax.set_ylabel(E_LABEL)
    y_ticks, y_tick_strs = generate_linear_ticks(E_range, NUM_TICKS, E_ROUNDING)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_strs)
    ax.set_ylim([E_min, E_max])

def format_I_yaxis(ax, I_range):

    I_min, I_max = I_range
    ax.set_ylabel(E_LABEL)
    y_ticks, y_tick_strs = generate_log_ticks(I_range, NUM_TICKS)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_strs)
    ax.set_ylim([I_min, I_max])

def plot_anisotropy_EvQ(AR, q_range=Q_RANGE_DEFAULT, E_range=E_RANGE_DEFAULT):

    q_min, q_max = q_range
    E_min, E_max = E_range

    AR_scale = np.nanmax([
        abs(np.nanpercentile(
            AR.sel(energy=slice(E_min,E_max)), 1
        )),
        abs(np.nanpercentile(
            AR.sel(energy=slice(E_min,E_max)), 99
        ))
    ])

    fig, ax = setup_plot()

    im = AR.sel(energy=slice(E_min,E_max)).plot(
        x='q', y='energy',
        ax=ax,
        vmin=-AR_scale,
        vmax=AR_scale,
        cmap=ANISOTROPY_CMAP,
        add_colorbar=False
    )

    add_cbar(fig, im, label='Anistropy')

    format_q_xaxis(ax, q_range)
    format_E_yaxis(ax, E_range)
    
    plt.show()
    # return fig

def plot_intensity_EvQ(I, q_range=Q_RANGE_DEFAULT, E_range=E_RANGE_DEFAULT, label=''):

    q_min, q_max = q_range
    E_min, E_max = E_range

    fig, ax = setup_plot()

    im = I.plot(
        x='q', y='energy',
        ax=ax,
        norm=LogNorm(1, np.nanpercentile(I, 97.5)),
        cmap=INTENSITY_CMAP,
        add_colorbar=False
    )

    add_cbar(fig, im)

    format_q_xaxis(ax, q_range)
    format_E_yaxis(ax, E_range)
    add_text(ax, label)

    plt.show()


    return

def plot_ISI(ISI):
    return

def plot_para_perp(para, perp, q_range=Q_RANGE_DEFAULT, I_range=I_RANGE_DEFAULT, E_range=E_RANGE_DEFAULT): 

    def update_plot(energy):

        ax.clear()
        para.sel(energy=energy).plot(ax=ax, yscale='log', xscale='linear', color='r')
        perp.sel(energy=energy).plot(ax=ax, yscale='log', xscale='linear', color='b')
        format_q_xaxis(ax, q_range)
        format_I_yaxis(ax, I_range)
        ax.legend(['Para', 'Perp'], fontsize=15, frameon=False)

    E_min, E_max = E_range

    fig, ax = setup_plot(figsize=(7,7))
    

    # energies = range(min(E_range), max(E_range))
    energies = para.sel(energy=slice(E_min,E_max)).energy
    print(energies)
    ani = FuncAnimation(fig, update_plot, frames=energies, repeat=False)

    plt.tight_layout()
    ani.save('paraperp.mp4', writer='ffmpeg', fps=5, dpi=300)
    plt.show()

    return

def setup_plot(figsize=(3.5,3.5)):

    plt.rcParams.update({
        "font.size": 18,
        "axes.linewidth": 2,
    })

    fig, ax = plt.subplots(1, 1, figsize = figsize)

    # Edit the major and minor ticks of the x and y axes
    ax.xaxis.set_tick_params(which = 'major', size = 5, width = 2, direction = 'in', top = 'on')
    ax.xaxis.set_tick_params(which = 'minor', size = 2.5, width = 2, direction = 'in', top = 'on')
    ax.yaxis.set_tick_params(which = 'major', size = 5, width = 2, direction = 'in', right = 'on')
    ax.yaxis.set_tick_params(which = 'minor', size = 2.5, width = 2, direction = 'in', right = 'on')
    
    return fig, ax

def save_fig(fig, save_dir='', filename='test', dpi=300):
    save_path = os.path.join(save_dir, filename + '.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', transparent=False)

