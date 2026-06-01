import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

toml_name = 'proximab.toml'
fix_name = 'proximab_fix.toml'
dirname = 'autotides_proxb_highXUV'
rootdir = '/Users/mvidaurr/Desktop/PycharmProjects/MO_tides/'
xuv_model = 1

# start_time = time.time()
ecc_range = np.linspace(0.02, 0.6, num=20)
spin_range = np.linspace(start=-100,stop=100,num=20)
time_range = [1e3, 1e5, 5e5, 1e6, 1e8, 5e8, 1e9]

def edit_toml(tomlfile='erf.toml'):
    with open('run_model.py', 'r+') as f:
        str = f.readlines()
    str[44] = 'simulation_config = \'' + (tomlfile) + '\'\n'
    with open('run_model.py', 'w') as f:
        f.writelines(str)

def edit_xuv(xuv=1):
    xuv_toml = toml_name
    trash_xuv = fix_name
    content = []
    with open(toml_name, 'r') as f:
        for line in f:
            content.append(line)

    new = []
    for c in content:
        if c.startswith('xuv_model'):
            new_xuv_string = (c[0:12]+'{:.0f}').format(xuv) + '\n'
            new.append(new_xuv_string)
        else:
            new.append(c)

    with open(fix_name, 'w') as f:
        for c in new:
            f.write(c)
    shutil.copyfile(trash_xuv, xuv_toml)
    os.remove(trash_xuv)

def edit_eccentricity(ecc=0.1):
    ecc_toml = toml_name
    trash_ecc = fix_name
    content = []
    with open(toml_name,'r') as f:
        for line in f:
            content.append(line)

    new = []
    for c in content:
        if c.startswith('initial_eccentricity'):
            new_ecc_string = (c[0:23] + '{:.2f}').format(ecc) + '\n'
            new.append(new_ecc_string)
        else:
            new.append(c)

    with open(fix_name,'w') as f:
        for c in new:
            f.write(c)
    shutil.copyfile(trash_ecc,ecc_toml)
    os.remove(trash_ecc)

def edit_spin(spin=0.0167):
    spin_toml = toml_name
    trash_spin = fix_name
    content = []
    with open(toml_name,'r') as f:
        for line in f:
            content.append(line)

    new = []
    for c in content:
        if c.startswith('initial_spin_multiplier'):
            new_spin_string = (c[0:26]+'{:.2f}').format(spin)+'\n'
            new.append(new_spin_string)
        else:
            new.append(c)

    with open(fix_name,'w') as f:
        for c in new:
            f.write(c)
    shutil.copyfile(trash_spin,spin_toml)
    os.remove(trash_spin)

def edit_time(time=1e9):
    time_toml = toml_name
    trash_time = fix_name
    content = []
    with open(toml_name,'r') as f:
        for line in f:
            content.append(line)

    new = []
    for c in content:
        if c.startswith('end_time_years'):
            new_time_string = (c[0:17]+'{:.0e}'+c[24:]).format(time)+'\n'
            new.append(new_time_string)
        else:
            new.append(c)

    with open(fix_name,'w') as f:
        for c in new:
            f.write(c)
    shutil.copyfile(trash_time,time_toml)
    os.remove(trash_time)

def edit_output(output='output.txt'):
    with open('run_model.py', 'r+') as f:
        str = f.readlines()
    str[408] = '    ' + (output) + '\n'
    with open('run_model.py', 'w') as f:
        f.writelines(str)

def create_dir():
    for t in time_range:
        os.makedirs('results/'+dirname+'/t={:.0e}'.format(t), exist_ok=True)
        edit_time(t)
    path = glob.glob('results/'+dirname+'/*')
    t_paths = []
    for p in path:
        t_paths.append(p)
    ecpaths = []
    for i in t_paths:
        name = i + '/ecc={:.3}'
        ecpaths.append(name)
        for ecc in ecc_range:
            os.makedirs(name.format(ecc), exist_ok=True)
            edit_eccentricity(ecc)
    path2 = glob.glob('results/'+dirname+'/*/*')
    s_paths = []
    for p in path2:
        s_paths.append(p)
    sppaths = []
    for i in s_paths:
        name = i + '/spin={:.3}'
        sppaths.append(name)
        for spin in spin_range:
            os.makedirs(name.format(float(spin)), exist_ok=True)
            edit_eccentricity(spin)

def autorun():
    create_dir()
    edit_toml(toml_name)
    edit_xuv(xuv_model)
    for time in time_range:
        edit_time(time)
        print('Time = ' + '{:.0e}'.format(time))
        for ecc in ecc_range:
            edit_eccentricity(ecc)
            print('Eccentricity = ' + '{:.3}'.format(ecc))
            for spin in spin_range:
                edit_spin(spin)
                print('Spin = ' + '{:.3}'.format(spin))
                edit_output('df_results.to_csv(f\'{save_name}_output_tm' + '{:.0e}'.format(time) +
                            '_ec' + '{:.3}'.format(ecc)
                            + '_sp' + '{:.3}'.format(spin) + '.txt' +
                            '\'' + ',' + ' sep=\',\',index=False)\n')
                os.system('python3 run_model.py')
                main_files = glob.glob('results/'+dirname+'/*/*/*')

                dest_png = []
                dest_txt = []
                for fname in main_files:
                    time_to_str = '{:.0e}'.format(time)
                    ecc_to_str = '{:.3}'.format(ecc)
                    spin_to_str = '{:.3}'.format(spin)
                    newpath = 'results/'+dirname+'/t=' + time_to_str + '/ecc=' + ecc_to_str + '/spin=' + spin_to_str
                    dest_png.append(newpath)
                    dest_txt.append(newpath)

                source_png = []
                source_txt = []
                mainf = os.listdir(rootdir)
                for pngfile in mainf:
                    if pngfile.endswith('.png'):
                        source_png.append(pngfile)
                searchstring = 'output'
                for outfile in mainf:
                    if searchstring in outfile:
                        source_txt.append(outfile)
                for (filep, destp) in zip(source_png, dest_png):
                    shutil.move(filep, destp)
                for (destt, filet) in zip(dest_txt, source_txt):
                    shutil.move(filet, destt)
                print('---------------------------- done! ---------------------------\n\n\n')

##--------------------- create dirs + run each case and save results ------------------------
# autorun()

##---------------------------------- make contour plots -------------------------------------
###may want to use rescale=True at the end of griddata for earth
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
    fig, ax = plt.subplots()
    time0 = '1e+03'
    for file in paths:
        if time0 in file:
                df1 = pd.read_csv(file)
                # print(file)
                df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'], 'PO2': df1['PO2']})
                df.dropna(axis=0, how='all', inplace=True)

                x = np.linspace(df['ecc'].min(), df['ecc'].max(), len(df['ecc'].unique()))
                y = np.linspace(df['spin'].min(), df['spin'].max(), len(df['spin'].unique()))
                z = griddata((df['ecc'], df['spin']), df['PO2'], (x[None, :], y[:, None]), method='nearest')
                plt.contourf(x, y, z)
    plt.colorbar(label='PO$_2$')
    plt.title(time0)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time1():
    fig, ax = plt.subplots()
    time1 = '1e+05'
    for file in paths:
        # print(file)
        if time1 in file:
            # print(file)
            df1 = pd.read_csv(file,sep=',',skipinitialspace=True)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'], 'PO2': df1['PO2']})
            # df.dropna(axis=0, how='all', inplace=True)
            #
            # x = np.linspace(df['ecc'].min(), df['ecc'].max(), len(df['ecc'].unique()))
            # y = np.linspace(df['spin'].min(), df['spin'].max(), len(df['spin'].unique()))
            plt.tricontourf(df['ecc'], df['spin'], df['PO2'])
    plt.colorbar(label='PO$_2$')
    plt.title(time1)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time2():
    fig, ax = plt.subplots()
    time2 = '1e+06'
    for file in paths:
        if time2 in file:
            df1 = pd.read_csv(file)
            # #print(file)
    #         df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'], 'PO2': df1['PO2']})
    #         df.dropna(axis=0, how='all', inplace=True)
    #
    #         x = np.linspace(df['ecc'].min(), df['ecc'].max(), len(df['ecc'].unique()))
    #         y = np.linspace(df['spin'].min(), df['spin'].max(), len(df['spin'].unique()))
    #         z = griddata((df['ecc'], df['spin']), df['PO2'], (x[None, :], y[:, None]), method='linear')
    #         plt.contourf(x, y, z)
    # plt.colorbar(label='PO$_2$')
    # plt.title(time2)
    # plt.xlabel('Eccentricity')
    # plt.ylabel('Spin')
    # plt.show()

def time3():
    fig, ax = plt.subplots()
    time3 = '1e+08'
    for file in paths:
        if time3 in file:
            df1 = pd.read_csv(file)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'], 'PO2': df1['PO2']})
            df.dropna(axis=0, how='all', inplace=True)

            x = np.linspace(df['ecc'].min(), df['ecc'].max(), len(df['ecc'].unique()))
            y = np.linspace(df['spin'].min(), df['spin'].max(), len(df['spin'].unique()))
            z = griddata((df['ecc'], df['spin']), df['PO2'], (x[None, :], y[:, None]), method='linear')
            plt.contourf(x, y, z)
    plt.colorbar(label='PO$_2$')
    plt.title(time3)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time4():
    fig, ax = plt.subplots()
    time4 = '1e+09'
    for file in paths:
        if time4 in file:
            df1 = pd.read_csv(file)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'], 'PO2': df1['PO2']})
            df.dropna(axis=0, how='all', inplace=True)

            x = np.linspace(df['ecc'].min(), df['ecc'].max(), len(df['ecc'].unique()))
            y = np.linspace(df['spin'].min(), df['spin'].max(), len(df['spin'].unique()))
            z = griddata((df['ecc'], df['spin']), df['PO2'], (x[None, :], y[:, None]), method='linear')
            plt.contourf(x, y, z)
    plt.colorbar(label='PO$_2$')
    plt.title(time4)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time5():
    fig, ax = plt.subplots()
    time5 = '5e+05'
    for file in paths:
        if time5 in file:
            df1 = pd.read_csv(file)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'], 'PO2': df1['PO2']})
            df.dropna(axis=0, how='all', inplace=True)

            x = np.linspace(df['ecc'].min(), df['ecc'].max(), len(df['ecc'].unique()))
            y = np.linspace(df['spin'].min(), df['spin'].max(), len(df['spin'].unique()))
            z = griddata((df['ecc'], df['spin']), df['PO2'], (x[None, :], y[:, None]), method='linear')
            plt.contourf(x, y, z)
    plt.colorbar(label='PO$_2$')
    plt.title(time5)
    plt.xlabel('Eccentricity')
    plt.ylabel('Spin')
    plt.show()

def time6():
    fig, ax = plt.subplots()
    time6 = '5e+08'
    for file in paths:
        if time6 in file:
            df1 = pd.read_csv(file)
            df = pd.DataFrame({'ecc': df1['eccentricity'], 'spin': df1['spin_ratio_planet'], 'PO2': df1['PO2']})
            df.dropna(axis=0, how='all', inplace=True)

            x = np.linspace(df['ecc'].min(), df['ecc'].max(), len(df['ecc'].unique()))
            y = np.linspace(df['spin'].min(), df['spin'].max(), len(df['spin'].unique()))
            z = griddata((df['ecc'], df['spin']), df['PO2'], (x[None, :], y[:, None]), method='linear')
            plt.contourf(x, y, z)
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