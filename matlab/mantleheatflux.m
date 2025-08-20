function [qm,Db,uc,Ra,nu] = mantleheatflux(Tm,Ts,rs,Rp,Rc,g,rho_m,FH2O,meltfrac)
%This function calculates the mantle heat flux, which depends on the
%temperature and viscosity of the mantle. The viscosity depends on
%temperature and is calculated with the function viscosity.m

%mantle thermal conductivity (W/m/K)
km = 4.2;
%depth of the convective zone
% if rs < Rp
if meltfrac >=0.4
    Z = Rp - rs; %convection only occurs within the magma ocean until solidification
else
    Z = Rp - Rc;
end

% coefficient of thermal expansion (1/K)
alpha = 2e-5;
%mantle heat capacity (J/kg/K)
cp = 1.2e3; 
%mantle thermal diffusivity (m^2 / s)
kappa = km / rho_m / cp;

%find the viscosity (m^2/s)
% if Ts < 1420
%     nu = viscosity2(Tm,FH2O,g,rho_m,0);
% else
    nu = viscosity(Tm,Ts,rho_m, FH2O, meltfrac);
% end

%Calculate Rayleigh number
Ra = (g * alpha * abs(Tm - Ts) * Z^3) / nu / kappa;

% %boundary layer depth (m)
% Db = Z * (Ra_cr / Ra)^beta;
qm = 0.089 * km * (Tm - Ts) * Ra^(1/3) / Z;
Db = km * (Tm - Ts) / qm;
    
%spreading time (s)
ts = Db^2 / (5.38 * kappa);%s

%spreading velocity (m/s)
uc = Z/ts; %m/s

end

