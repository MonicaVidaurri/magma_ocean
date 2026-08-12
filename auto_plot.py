import glob
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.ticker as ticker

dirname = 'autotides_t1e_lowXUV_tidesoff'
plotdir = 'results/plots/new_PO2/tidesOFF/t1e_high'

rootdir = '/Users/mvidaurr/Desktop/PycharmProjects/magma_ocean'
# ecc_range = np.linspace(0.02, 0.6, num=20)
# spin_range = np.linspace(start=-100,stop=100,num=20)
# time_range = [1e3, 1e5, 5e5, 1e6, 1e8, 5e8, 1e9]

# ecc_range = np.linspace(0.02, 0.6, num=5)
# spin_range = np.linspace(start=-100,stop=100,num=15)
time_range = [1e5, 1e6, 1e7, 1e8, 1e9]

ecc_range = np.linspace(0.0001, 0.2, 20)
spin_range = np.arange(-10,11,dtype=np.float64)

results_path = glob.glob('results/low_values/'+dirname+'/*/*/*')
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
    dfs_all = []
    for file in paths:
        if time0 in file:
            df1 = pd.read_csv(file, sep=",", skipinitialspace=True)
            temp_df = pd.DataFrame({"ecc": df1["eccentricity"],"spin": df1["spin_ratio_planet"],"PO2": df1["PO2"]})
            temp_df.dropna(inplace=True)
            dfs_all.append(temp_df)

    df_all = pd.concat(dfs_all, ignore_index=True)

    x = df_all['ecc'].values
    y = df_all['spin'].values
    z = df_all['PO2'].values

    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = np.linspace(y.min(), y.max(), 200)
    x1, y1 = np.meshgrid(x_grid, y_grid)

    z1 = griddata((x, y), z, (x1, y1), method='linear')

    contour = plt.contourf(x1,y1,z1, levels=20, cmap='viridis')
    ####colorbar type 1
    # cbar = plt.colorbar(contour)
    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    # cbar.set_label('PO2')
    ####colorbar type 2
    tick_locations = np.linspace(df_all['PO2'].min(), df_all['PO2'].max(), 7)
    cbar = plt.colorbar(contour,ticks=tick_locations)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.ax.yaxis.set_ticklabels(['1e-56', '1e5', '5e5', '1e6', '5e6', '1e7', '1e8'])
    cbar.set_label('PO2')

    #overlay OG data points to see the density
    # plt.scatter(x, y, color='black', s=5, alpha=0.3, label='Data Points')

    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.title('t = ' + time0 + ' years')
    plt.savefig(plotdir+'/t_'+time0+'.png')

    plt.show()

def time1():
    time1 = '1e+06'
    dfs_all = []

    for file in paths:
        if time1 in file:
            df1 = pd.read_csv(file, sep=",", skipinitialspace=True)
            temp_df = pd.DataFrame({"ecc": df1["eccentricity"],"spin": df1["spin_ratio_planet"],"PO2": df1["PO2"]})
            temp_df.dropna(inplace=True)
            dfs_all.append(temp_df)

    df_all = pd.concat(dfs_all, ignore_index=True)

    x = df_all['ecc'].values
    y = df_all['spin'].values
    z = df_all['PO2'].values

    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = np.linspace(y.min(), y.max(), 200)
    x1, y1 = np.meshgrid(x_grid, y_grid)

    z1 = griddata((x, y), z, (x1, y1), method='linear')

    contour = plt.contourf(x1,y1,z1, levels=20, cmap='viridis')
    ####colorbar type 1
    # cbar = plt.colorbar(contour)
    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    # cbar.set_label('PO2')
    ####colorbar type 2
    tick_locations = np.linspace(df_all['PO2'].min(), df_all['PO2'].max(), 7)
    cbar = plt.colorbar(contour,ticks=tick_locations)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.ax.yaxis.set_ticklabels(['1e-56', '1e5', '5e5', '1e6', '5e6', '1e7', '1e8'])
    cbar.set_label('PO2')

    #overlay OG data points to see the density
    # plt.scatter(x, y, color='black', s=5, alpha=0.3, label='Data Points')

    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.title('t = ' + time1 + ' years')
    plt.savefig(plotdir+'/t_'+time1+'.png')

    plt.show()

def time2():
    time2 = '1e+07'
    dfs_all = []

    for file in paths:
        if time2 in file:
            df1 = pd.read_csv(file, sep=",", skipinitialspace=True)
            temp_df = pd.DataFrame({"ecc": df1["eccentricity"], "spin": df1["spin_ratio_planet"], "PO2": df1["PO2"]})
            temp_df.dropna(inplace=True)
            dfs_all.append(temp_df)

    df_all = pd.concat(dfs_all, ignore_index=True)

    x = df_all['ecc'].values
    y = df_all['spin'].values
    z = df_all['PO2'].values

    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = np.linspace(y.min(), y.max(), 200)
    x1, y1 = np.meshgrid(x_grid, y_grid)

    z1 = griddata((x, y), z, (x1, y1), method='linear')

    contour = plt.contourf(x1, y1, z1, levels=20, cmap='viridis')
    ####colorbar type 1
    # cbar = plt.colorbar(contour)
    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    # cbar.set_label('PO2')
    ####colorbar type 2
    tick_locations = np.linspace(df_all['PO2'].min(), df_all['PO2'].max(), 7)
    cbar = plt.colorbar(contour,ticks=tick_locations)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.ax.yaxis.set_ticklabels(['1e-56', '1e5', '5e5', '1e6', '5e6', '1e7', '1e8'])
    cbar.set_label('PO2')

    # overlay OG data points to see the density
    # plt.scatter(x, y, color='black', s=5, alpha=0.3, label='Data Points')

    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.title('t = ' + time2 + ' years')
    plt.savefig(plotdir + '/t_' + time2 + '.png')

    plt.show()

def time3():
    time3 = '1e+08'
    dfs_all = []

    for file in paths:
        if time3 in file:
            df1 = pd.read_csv(file, sep=",", skipinitialspace=True)
            temp_df = pd.DataFrame({"ecc": df1["eccentricity"], "spin": df1["spin_ratio_planet"], "PO2": df1["PO2"]})
            temp_df.dropna(inplace=True)
            dfs_all.append(temp_df)

    df_all = pd.concat(dfs_all, ignore_index=True)

    x = df_all['ecc'].values
    y = df_all['spin'].values
    z = df_all['PO2'].values

    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = np.linspace(y.min(), y.max(), 200)
    x1, y1 = np.meshgrid(x_grid, y_grid)

    z1 = griddata((x, y), z, (x1, y1), method='linear')

    contour = plt.contourf(x1, y1, z1, levels=20, cmap='viridis')
    ####colorbar type 1
    # cbar = plt.colorbar(contour)
    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    # cbar.set_label('PO2')
    ####colorbar type 2
    tick_locations = np.linspace(df_all['PO2'].min(), df_all['PO2'].max(), 7)
    cbar = plt.colorbar(contour,ticks=tick_locations)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.ax.yaxis.set_ticklabels(['1e-56', '1e5', '5e5', '1e6', '5e6', '1e7', '1e8'])
    cbar.set_label('PO2')

    # overlay OG data points to see the density
    # plt.scatter(x, y, color='black', s=5, alpha=0.3, label='Data Points')

    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.title('t = ' + time3 + ' years')
    plt.savefig(plotdir + '/t_' + time3 + '.png')

    plt.show()

def time4():
    time4 = '1e+09'
    dfs_all = []

    for file in paths:
        if time4 in file:
            df1 = pd.read_csv(file, sep=",", skipinitialspace=True)
            temp_df = pd.DataFrame({"ecc": df1["eccentricity"], "spin": df1["spin_ratio_planet"], "PO2": df1["PO2"]})
            temp_df.dropna(inplace=True)
            dfs_all.append(temp_df)

    df_all = pd.concat(dfs_all, ignore_index=True)
    x = df_all['ecc'].values
    y = df_all['spin'].values
    z = df_all['PO2'].values

    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = np.linspace(y.min(), y.max(), 200)
    x1, y1 = np.meshgrid(x_grid, y_grid)

    z1 = griddata((x, y), z, (x1, y1), method='linear')
    contour = plt.contourf(x1, y1, z1, levels=20, cmap='viridis')
    ####colorbar type 1
    # cbar = plt.colorbar(contour)
    # cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    # cbar.set_label('PO2')
    ####colorbar type 2
    tick_locations = np.linspace(df_all['PO2'].min(), df_all['PO2'].max(), 7)
    cbar = plt.colorbar(contour,ticks=tick_locations)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.ax.yaxis.set_ticklabels(['1e-56', '1e5', '5e5', '1e6', '5e6', '1e7', '1e8'])
    cbar.set_label('PO2')

    # overlay OG data points to see the density
    # plt.scatter(x, y, color='black', s=5, alpha=0.3, label='Data Points')

    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.title('t = ' + time4 + ' years')
    plt.savefig(plotdir + '/t_' + time4 + '.png')

    plt.show()

# time0()
# time1()
# time2()
# time3()
time4()