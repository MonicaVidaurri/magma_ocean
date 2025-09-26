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

##### TRAPPIST-1; all data from Agol et al. 2021 #####
MStar = 0.0898 * MSun
LStar = 0.000553 * LSun

##### comment out every block except the planet you wanna use! #####
##### 1 c #####
# a = 1.58e-2*AU
# Mp = 1.308*MEarth
# Rp = 1.097*REarth
# Rc = 0.518*Rp
# core_fraction= 0.266

##### 1 d #####
# a = 2.227e-2*AU
# Mp = 0.388*MEarth
# Rp = 0.788*REarth
# Rc = 0.426*Rp
# core_fraction = 0.177

##### 1 e #####
a = 2.925e-2*AU
Mp = 0.692*MEarth
Rp = 0.92*REarth
Rc = 0.467*Rp
core_fraction = 0.236

##### 1 f #####
# a = 3.849e-2*AU
# Mp = 1.039*MEarth
# Rp = 1.045*REarth
# Rc = 0.429*Rp
# core_fraction = 0.192

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