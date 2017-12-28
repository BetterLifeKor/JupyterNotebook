# Day_03_01_colormap.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as color


def color_1():
    x = np.random.rand(100)
    y = np.random.rand(100)
    t = np.arange(100)

    plt.scatter(x, y, c=t)
    plt.show()


def color_2():
    x = np.arange(100)
    y = x
    t = x

    plt.scatter(x, y, c=t)
    # plt.scatter(x, -y, c=t, cmap='viridis')
    # plt.scatter(x, -y, c=[cm.viridis(0), cm.viridis(255)])
    plt.scatter(x, -y, c=[cm.viridis(0)] * 50 +
                         [cm.viridis(255)] * 50)
    plt.show()


def color_3():
    print(plt.colormaps())
    # ['Accent', 'Accent_r', 'Blues', 'Blues_r',
    # 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
    # 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r',
    # 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', ...]

    x = np.arange(100)
    y = x
    t = x

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(x, y, c=t, cmap='viridis')
    # ax2.scatter(x, y, c=t, cmap='viridis_r')
    # ax2.scatter(x, y, c=-t, cmap='viridis')
    ax2.scatter(x, y, c=t[::-1], cmap='viridis')
    plt.show()

    x = np.arange(50)
    y = x
    t = x

    plt.scatter(x, y, c=t)
    plt.colorbar()
    plt.show()


def color_4():
    values = np.random.rand(10, 10)
    print(values)
    print(values.shape)
    plt.imshow(values)
    plt.copper()
    # plt.zet()             # 제공 안함.
    plt.show()


def color_5():
    jet = cm.get_cmap('jet')
    print(jet)

    print(jet(-10))
    print(jet(0))
    print(jet(127))
    print(jet(255))
    print(jet(2550))
    print('-' * 50)

    print(jet(0.1))
    print(jet(0.2))
    print(jet(0.3))
    print('-' * 50)

    print(jet([0, 255]))
    print(jet(range(0, 5, 2)))
    print(jet(np.linspace(0.3, 0.7, 5)))


def color_6():
    # Have colormaps separated into categories:
    # http://matplotlib.org/examples/color/colormaps_reference.html
    cmaps = [('Perceptually Uniform Sequential', [
        'viridis', 'plasma', 'inferno', 'magma']),
             ('Sequential', [
                 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
             ('Sequential (2)', [
                 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                 'hot', 'afmhot', 'gist_heat', 'copper']),
             ('Diverging', [
                 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
             ('Qualitative', [
                 'Pastel1', 'Pastel2', 'Paired', 'Accent',
                 'Dark2', 'Set1', 'Set2', 'Set3',
                 'tab10', 'tab20', 'tab20b', 'tab20c']),
             ('Miscellaneous', [
                 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

    nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(cmap_category, cmap_list, nrows):
        fig, axes = plt.subplots(nrows=nrows)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
            ax.set_axis_off()

    for cmap_category, cmap_list in cmaps:
        plot_color_gradients(cmap_category, cmap_list, nrows)

    plt.show()


color_6()



