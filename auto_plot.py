import glob
import matplotlib.pyplot as plt
import numpy as np

dirname = 'autotides_proxb_highXUV'
rootdir = '/Users/mvidaurr/Desktop/PycharmProjects/MO_tides/'
ecc_range = np.linspace(0.02, 0.6, num=20)
spin_range = np.linspace(start=-100,stop=100,num=20)
time_range = [1e3, 1e5, 5e5, 1e6, 1e8, 5e8, 1e9]

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
    time0 = '1e+03'
    for file in paths:
        # print(file)
        if time0 in file:
            # print(file)
            df = np.genfromtxt(file, skip_header=1, delimiter=',')
            plt.tricontourf(df[:, 3], df[:, 5], df[:, 13])
    plt.colorbar(label='PO$_2$')
    plt.title(time0)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time1():
    time1 = '1e+05'
    for file in paths:
        # print(file)
        if time1 in file:
            # print(file)
            df = np.genfromtxt(file, skip_header=1, delimiter=',')
            plt.tricontourf(df[:,3], df[:,5], df[:,13])
    plt.colorbar(label='PO$_2$')
    plt.title(time1)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time2():
    time2 = '1e+06'
    for file in paths:
        # print(file)
        if time2 in file:
            # print(file)
            df = np.genfromtxt(file, skip_header=1, delimiter=',')
            plt.tricontourf(df[:, 3], df[:, 5], df[:, 13])
    plt.colorbar(label='PO$_2$')
    plt.title(time2)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time3():
    time3 = '1e+08'
    for file in paths:
        # print(file)
        if time3 in file:
            # print(file)
            df = np.genfromtxt(file, skip_header=1, delimiter=',')
            plt.tricontourf(df[:, 3], df[:, 5], df[:, 13])
    plt.colorbar(label='PO$_2$')
    plt.title(time3)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time4():
    time4 = '1e+09'
    for file in paths:
        # print(file)
        if time4 in file:
            # print(file)
            df = np.genfromtxt(file, skip_header=1, delimiter=',')
            plt.tricontourf(df[:, 3], df[:, 5], df[:, 13])
    plt.colorbar(label='PO$_2$')
    plt.title(time4)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time5():
    time5 = '5e+05'
    for file in paths:
        # print(file)
        if time5 in file:
            # print(file)
            df = np.genfromtxt(file, skip_header=1, delimiter=',')
            plt.tricontourf(df[:, 3], df[:, 5], df[:, 13])
    plt.colorbar(label='PO$_2$')
    plt.title(time5)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time6():
    time6 = '5e+08'
    for file in paths:
        # print(file)
        if time6 in file:
            # print(file)
            df = np.genfromtxt(file, skip_header=1, delimiter=',')
            plt.tricontourf(df[:, 3], df[:, 5], df[:, 13])
    plt.colorbar(label='PO$_2$')
    plt.title(time6)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

# time0()
time1()
# time2()
# time3()
# time4()
# time5()
# time6()