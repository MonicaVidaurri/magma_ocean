% load stellar_dataProxCen.mat
data = 'proxcen_stellardata.xlsx'
t = xlsread(data,'A:A');
L_GJ = xlsread(data,'B:B');

save('stellar_dataProxCen.mat','t','L_GJ')