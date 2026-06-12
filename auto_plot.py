import glob
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.ticker as ticker

dirname = 'autotides_proxb_lowXUV'
rootdir = '/Users/mvidaurr/Desktop/PycharmProjects/magma_ocean'
# ecc_range = np.linspace(0.02, 0.6, num=20)
# spin_range = np.linspace(start=-100,stop=100,num=20)
# time_range = [1e3, 1e5, 5e5, 1e6, 1e8, 5e8, 1e9]

ecc_range = np.linspace(0.02, 0.6, num=5)
spin_range = np.linspace(start=-100,stop=100,num=15)
time_range = [1e5, 1e6, 1e7, 1e8, 1e9]

results_path = glob.glob('results/'+dirname+'/*/*/*')
paths = []
ecc_vals = []
spin_vals = []
for t in time_range:
    new_time = '{:.0e}'.format(t)
    for e in ecc_range:
        new_ecc = '{:.3}'.format(e)
        ecc_vals.append(new_ecc)
        for s in spin_range:
            new_spin = '{:.3}'.format(s)
            spin_vals.append(new_spin)
            srchstring = 'output'
            for path in results_path:
                pathstr = 'results/'+dirname+'/t=' + new_time + '/ecc=' + new_ecc + '/spin=' + new_spin
                use_paths = glob.glob(pathstr+'/*')
            for file in use_paths:
                if srchstring in file:
                    paths.append(file)

def time0():
    time0 = '1e+05'
    all_dfs = []
    for file in paths:
        # print(file)
        if time0 in file:
            # print(file)
            df1 = pd.read_csv(file, sep=',',skipinitialspace=True)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'],
                               'PO2': df1['PO2']})
            all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    x_min, x_max = df['ecc'].min(), df['ecc'].max()
    y_min, y_max = df['spin'].min(), df['spin'].max()
    x_norm = (df['ecc'] - x_min) / (x_max - x_min)
    y_norm = (df['spin'] - y_min) / (y_max - y_min)
    x_grid_norm = np.linspace(0, 1, 200)
    y_grid_norm = np.linspace(0, 1, 200)
    X_norm, Y_norm = np.meshgrid(x_grid_norm, y_grid_norm)

    Z = griddata(
        (x_norm, y_norm),
        df['PO2'],(X_norm, Y_norm),method='linear')
    X_orig = X_norm * (x_max - x_min) + x_min
    Y_orig = Y_norm * (y_max - y_min) + y_min
    my_levels = np.logspace(np.log10(1), np.log10(1e8), num=20)
    contour = plt.contourf(X_orig, Y_orig, Z, cmap='coolwarm', levels=my_levels, norm=colors.LogNorm(vmin=1, vmax=1e8))
    cbar = plt.colorbar(contour, label='PO2', format='%.0e')
    cbar.locator = ticker.LogLocator(base=10.0, numticks=12)
    def clean_exponent(x, pos):
        return f"{x:.0e}".replace("+0", "").replace("+", "")
    cbar.formatter = ticker.FuncFormatter(clean_exponent)
    cbar.update_ticks()
    plt.title('t = ' + time0 + ' years')
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time1():
    time1 = '1e+06'
    all_dfs = []
    for file in paths:
        # print(file)
        if time1 in file:
            # print(file)
            df1 = pd.read_csv(file, sep=',',skipinitialspace=True)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'],
                               'PO2': df1['PO2']})
            all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    x_min, x_max = df['ecc'].min(), df['ecc'].max()
    y_min, y_max = df['spin'].min(), df['spin'].max()
    x_norm = (df['ecc'] - x_min) / (x_max - x_min)
    y_norm = (df['spin'] - y_min) / (y_max - y_min)
    x_grid_norm = np.linspace(0, 1, 200)
    y_grid_norm = np.linspace(0, 1, 200)
    X_norm, Y_norm = np.meshgrid(x_grid_norm, y_grid_norm)

    Z = griddata(
        (x_norm, y_norm),
        df['PO2'],(X_norm, Y_norm),method='linear')
    X_orig = X_norm * (x_max - x_min) + x_min
    Y_orig = Y_norm * (y_max - y_min) + y_min
    my_levels = np.logspace(np.log10(1), np.log10(1e8), num=20)
    contour = plt.contourf(X_orig, Y_orig, Z, cmap='coolwarm', levels=my_levels, norm=colors.LogNorm(vmin=1, vmax=1e8))
    cbar = plt.colorbar(contour, label='PO2', format='%.0e')
    cbar.locator = ticker.LogLocator(base=10.0, numticks=12)
    def clean_exponent(x, pos):
        return f"{x:.0e}".replace("+0", "").replace("+", "")
    cbar.formatter = ticker.FuncFormatter(clean_exponent)
    cbar.update_ticks()
    plt.title('t = ' + time1 + ' years')
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time2():
    time2 = '1e+07'
    all_dfs = []
    for file in paths:
        # print(file)
        if time2 in file:
            # print(file)
            df1 = pd.read_csv(file, sep=',',skipinitialspace=True)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'],
                               'PO2': df1['PO2']})
            all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    x_min, x_max = df['ecc'].min(), df['ecc'].max()
    y_min, y_max = df['spin'].min(), df['spin'].max()
    x_norm = (df['ecc'] - x_min) / (x_max - x_min)
    y_norm = (df['spin'] - y_min) / (y_max - y_min)
    x_grid_norm = np.linspace(0, 1, 200)
    y_grid_norm = np.linspace(0, 1, 200)
    X_norm, Y_norm = np.meshgrid(x_grid_norm, y_grid_norm)

    Z = griddata(
        (x_norm, y_norm),
        df['PO2'],(X_norm, Y_norm),method='linear')
    X_orig = X_norm * (x_max - x_min) + x_min
    Y_orig = Y_norm * (y_max - y_min) + y_min
    my_levels = np.logspace(np.log10(1), np.log10(1e8), num=20)
    contour = plt.contourf(X_orig, Y_orig, Z, cmap='coolwarm', levels=my_levels, norm=colors.LogNorm(vmin=1, vmax=1e8))
    cbar = plt.colorbar(contour, label='PO2', format='%.0e')
    cbar.locator = ticker.LogLocator(base=10.0, numticks=12)
    def clean_exponent(x, pos):
        return f"{x:.0e}".replace("+0", "").replace("+", "")
    cbar.formatter = ticker.FuncFormatter(clean_exponent)
    cbar.update_ticks()
    plt.title('t = ' + time2 + ' years')
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time3():
    time3 = '1e+08'
    all_dfs = []
    for file in paths:
        # print(file)
        if time3 in file:
            # print(file)
            df1 = pd.read_csv(file, sep=',',skipinitialspace=True)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'],
                               'PO2': df1['PO2']})
            all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    x_min, x_max = df['ecc'].min(), df['ecc'].max()
    y_min, y_max = df['spin'].min(), df['spin'].max()
    x_norm = (df['ecc'] - x_min) / (x_max - x_min)
    y_norm = (df['spin'] - y_min) / (y_max - y_min)
    x_grid_norm = np.linspace(0, 1, 200)
    y_grid_norm = np.linspace(0, 1, 200)
    X_norm, Y_norm = np.meshgrid(x_grid_norm, y_grid_norm)

    Z = griddata(
        (x_norm, y_norm),
        df['PO2'],(X_norm, Y_norm),method='linear')
    X_orig = X_norm * (x_max - x_min) + x_min
    Y_orig = Y_norm * (y_max - y_min) + y_min
    my_levels = np.logspace(np.log10(1), np.log10(1e8), num=20)
    contour = plt.contourf(X_orig, Y_orig, Z, cmap='coolwarm', levels=my_levels, norm=colors.LogNorm(vmin=1, vmax=1e8))
    cbar = plt.colorbar(contour, label='PO2', format='%.0e')
    cbar.locator = ticker.LogLocator(base=10.0, numticks=12)
    def clean_exponent(x, pos):
        return f"{x:.0e}".replace("+0", "").replace("+", "")
    cbar.formatter = ticker.FuncFormatter(clean_exponent)
    cbar.update_ticks()
    plt.title('t = ' + time3 + ' years')
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time4():
    time4 = '1e+09'
    all_dfs = []
    for file in paths:
        # print(file)
        if time4 in file:
            # print(file)
            df1 = pd.read_csv(file, sep=',',skipinitialspace=True)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'],
                               'PO2': df1['PO2']})
            all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    x_min, x_max = df['ecc'].min(), df['ecc'].max()
    y_min, y_max = df['spin'].min(), df['spin'].max()
    x_norm = (df['ecc'] - x_min) / (x_max - x_min)
    y_norm = (df['spin'] - y_min) / (y_max - y_min)
    x_grid_norm = np.linspace(0, 1, 200)
    y_grid_norm = np.linspace(0, 1, 200)
    X_norm, Y_norm = np.meshgrid(x_grid_norm, y_grid_norm)

    Z = griddata(
        (x_norm, y_norm),
        df['PO2'],(X_norm, Y_norm),method='linear')
    X_orig = X_norm * (x_max - x_min) + x_min
    Y_orig = Y_norm * (y_max - y_min) + y_min
    my_levels = np.logspace(np.log10(1), np.log10(1e8), num=20)
    contour = plt.contourf(X_orig, Y_orig, Z, cmap='coolwarm',levels=my_levels,norm=colors.LogNorm(vmin=1, vmax=1e8))
    cbar = plt.colorbar(contour, label='PO2',format='%.0e')
    cbar.locator = ticker.LogLocator(base=10.0, numticks=12)
    def clean_exponent(x, pos):
        return f"{x:.0e}".replace("+0", "").replace("+", "")
    cbar.formatter = ticker.FuncFormatter(clean_exponent)
    cbar.update_ticks()
    plt.title('t = ' + time4 + ' years')
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

time0()
time1()
time2()
time3()
time4()