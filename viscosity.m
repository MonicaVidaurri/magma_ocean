function nu = viscosity(Tm,Ts,rho_m,FH2O,meltfrac)
%This function calculates the viscosity of the magma ocean, following the
%parameterization of Lebrun et al. (2013). It depends on the mantle
%temperature (Tm) and the crystal fraction in the melt
rho_m = 3.3e3;
%liquid magma dynamic viscosity (Pa s)
A = 0.00024; %Pa s
B = 4600; %K
eta_l = A * exp(B/(Tm - 1000));

% %melt-fraction dependent dynamic viscosity (Pa s)
alphan = 26; 
% mu = 80; %GPa
% Am = 5.3e15; %pre-exponential factor
% h = 1; %mm
% b = 0.5; %nm
% n = 2.5; %exponent
% Ea = 240; %kJ/mol
% R = 8.31451; %ideal gas constant J/mole
% eta_s = mu / (2 * Am) * (h / (b*1e-6))^n * exp(Ea*1e3 / R / Tm);
eta_s = 3.7489e9 * exp(350e3/8.31447 / Tm);

% solid mantle kinematic viscosity McGovern & Schubert (1989)
alpha1 = 6.4e4;
alpha2 = -6.1e6;
nu0 = 2.21e7; %m2/s

meltfrac = (Tm - 1420)/600;
if Tm < 1420;
    meltfrac = 0;
end
if meltfrac > 1
    meltfrac = 1.0;
end


%viscosity of magma ocean behaves as a liquid for melt fractions greater
%than 40%, or as a solid for melt fractions smaller than 40%
if 1-meltfrac < 0.6
    eta = eta_l / ( 1 - (1 - meltfrac) / (1 - 0.4)) ^ (2.5);
    nu  = eta / rho_m;
else%if 1 - meltfrac < 0.99
    eta = eta_s * exp(-alphan * meltfrac);
    nu = eta / rho_m;
% else
%     nu = nu0 * exp((alpha1 + alpha2 * FH2O)/Tm);
end
    

end