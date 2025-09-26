### imports

##### constants #####
G = 6.67e-11 #m^3/kg/s^2
sigma = 5.67e-8 #W/m^2/K^4
### call stellar data, temp+pressure data, grids, etc
### UV model A or B
tsat = 1e9


##### stellar + Earth properties #####
MSun = 1.989e30 #kg
LSun = 3.846e26 #W
AU = 149597871*1e3 #Earth AU [m]
MEarth = 5.97e24 #kg
REarth = 6371e3 #m

##### Proxima b; data from Faria et al. 2022 #####
MStar = 0.1221 * MSun
LStar = 0.0016 * LSun
a = 4.8e-2*AU
Mp = 1.07*MEarth
Rp = 1.03*REarth
Rc = 0.429*Rp
core_fraction = 0.2

##### water abundance calculations #####
Mmantle = (1-core_fraction) * Mp
Earth_ocean = 1.39e21 #kg
#Teq calculated elsewhere; insert here
gp = G * Mp / Rp^2 #m/s^2
# # SET WATER ABUNDANCE HERE!!!!! replace the 2 # #
MH2O = 2 * Earth_ocean

FH2O = MH2O/Mmantle
FeOt = 0.08
Fe3_Fet = 1e-12

#######insert the rest of the code starting w/ solidus temp params...