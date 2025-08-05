% function [ Phi1crit ] = get_loss( t_flux,L_GJ,t,PO2,PH2O,model)
function [ phi_H,phi_O ] = get_loss( t_flux,L_GJ,t,PO2,PH2O,tsat,a,MPlanet,rPlanet,LStar)
%calculate the XUV driven atmospheric loss using the present atmospheric
%composition (Patm = H2O)
%% define parameters 
G       = 6.673e-11;                      % gravitational constant [m3/kg/s2]
MSun    = 1.989e30;                       % sun mass [kg]
LSun    = 3.846e26;                       % sun luminosity [W]
AU      = 149597871*1e3;                  % earth semimajor axis [m]
ME      = 5.972e24;                       % earth mass [kg]
rE      = 6371*1e3;                       % earth radius [m]
sigma   = 5.67e-8;                        % Stefan-Boltzmann constant [W/m2/K4]
Rsun = 695950.0e3;% m

Fat1AU  = LStar*1366;                     % flux from star at 1 AU [W/m2]
F       = Fat1AU*(AU/a)^2;                % flux from star at planet's orbit [W/m2]
A       = 0.25;                           % planet albedo []
g0      = G*MPlanet/rPlanet^2;            % planet surface gravity [m/s2]
ASR     = (1-A)*F/4;                      % absorbed stellar radiation [W/m2]
Teq     = (ASR/sigma)^0.25;               % equilibrium temperature [K]
muH     = 1.008;                       % atomic weight of hydrogen (kg/mol)
mpr     = 1.6726219e-27;                  % mass of a proton (kg)
muO     = 15.9994;                     % atomic weight of oxygen (kg/mol) 
kB      = 1.38064852e-23;                 % Boltzmann constant, J/K
NA      = 6.022e23;                       % Avogadro's number (molecules/mole)
muH2O = 18.015;
muO2 = muO * 2;


%% calculate XUV-driven hydrodynamic mass loss 
% model A: follow formula from Ribas+ (2005)
% model B: assume saturation phase for 1 Gy, then nothing

yr        = 60*60*24*365.25;         % 1 y [s]
f0        = 1e-3;                  % XUV saturation fraction
% tsat      = 1e8;                     % XUV saturation time; [yr]
beta      = -1.23;                   % decay constant
eff       = 0.1;                     % energy conversion efficiency factor []
f         = f0*(t_flux/tsat).^beta;        % XUV fraction vs. time

% if t_flux < tsat
%     f = f0;
%     L_XUV     = f.*L_GJ*LSun;            % XUV luminosity [W]
%     F_XUV     = L_XUV/(4*pi*a^2);        % XUV flux at GJ1132b orbit [W/m2]
% else
%     L_XUV     = f.*L_GJ*LSun;            % XUV luminosity [W]
%     F_XUV     = L_XUV/(4*pi*a^2);        % XUV flux at GJ1132b orbit [W/m2]
% end

% if(model==1) %XUV model A
    f(t_flux<tsat) = f0;
    L_XUV     = f.*L_GJ*LSun;            % XUV luminosity [W]
    F_XUV     = L_XUV/(4*pi*a^2);        % XUV flux at GJ1132b orbit [W/m2]
% else %XUV model B
%     f(t_flux<tsat) = f0;
%     f(t_flux>tsat) = 0.0;
%     L_XUV     = f.*L_GJ(end)*LSun;       % XUV luminosity [W]
%     F_XUV     = L_XUV/(4*pi*a^2);        % XUV flux at GJ1132b orbit [W/m2]
% end
Vpot       = G*MPlanet/rPlanet;                                  % gravitational potential [J/kg]

% Find XUV flux at present time
if t > t_flux(1)
    Fxuv = interp1(t_flux,F_XUV,t);
else
    Fxuv = F_XUV(1);
end


%% now calculate fractionation of the atmosphere

phi    = (eff*Fxuv/4)/Vpot;  % mass escape rate [W/m2 / J/kg = kg/m2/s]

% set up parameters
Tesc    = Teq;                 % temperature of escaping region [K]
Phi1ref = phi /(muH * mpr);         % total equivalent hydrogen escape rate [molec./m2/s]
mu1     = muH;                   % molar mass of species 1 [g/mol]
mu2     = muO;                   % molar mass of species 2 [g/mol]
% X1      = (PH2O * 2/muH2O) /(PH2O*(3/muH2O) + PO2*(2/muO2));% molar concentration of species H [mol/mol]
% X2      = 1.0 - X1;              % molar concentration of species 2 [mol/mol]
X1      = 2/3;                   % molar concentration of species 1 [mol/mol]
X2      = 1.0 - X1;              % molar concentration of species 2 [mol/mol]


% calculate crossover mass and species 1 and 2 fluxes
b            = 4.8e17*(Tesc)^0.75*1e2;                                % binary diffusion coeff [1/ms]; from Zahnle & Kasting (1986)
gam          = 1/(1 + X2*mu2/(X1*mu1));                               % should be 1/9 for H2O escape
mu_c_ref     = mu1 + kB*Tesc*Phi1ref./(b*g0*X1*mpr);                  % reference crossover mass [amu]
mu_c         = mu2 + gam*(mu_c_ref - mu2);                            % actual crossover mass [amu] (Chassefiere 1996b eqn. (9)
Phi1         = Phi1ref.*(mu_c./mu_c_ref);                             % flux of species 1 [molec./m2/s]
Phi2         = (X2/X1).*Phi1.*((mu_c - mu2)./(mu_c - mu1));           % flux of species 2 [molec./m2/s] (Luger & Barnes 2014)
Phi2(Phi2<0) = 0.0;

Phi1crit   = (b*g0*X1*mpr)*(mu2-mu1)/(kB*Tesc);                % critical flux of species 1 below which no more species 2 escape [molec./m2/s]

phi_H = Phi1 / NA * muH * 1e-3;
% phi_H = min([Phi1 Phi1crit]) / NA * muH * 1e-3;
if min([Phi1 Phi1crit]) == Phi1crit
    phi_O = Phi2 / NA * muO * 1e-3;
else
    phi_O = 0.0;
end

end

