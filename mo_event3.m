function [fun,isterminal,direction] = mo_event3(t,Tr);
%event detection for the magma ocean evolution
% event is when the planet surface reaches the solidus temperature and
% therefore radius of solidification reaches the surface
%constants
G = 6.67e-11;%m3/kg/s2
sigma = 5.67e-8; %W/m2/K4
% Star properties
MSun    = 1.989e30;                       % sun mass [kg]
LSun    = 3.846e26;                       % sun luminosity [W]
AU      = 149597871*1e3;                  % earth semimajor axis [m]
MStar   = 0.181;                          % mass of GJ1132 [MSun]
LStar   = 0.00438;                        % luminosity of GJ1132 [LSun]
Tday    = 1.628930;                       % orbital period [days]
Omega   = 2*pi/(Tday*86400);              % orbital frequency [rad/s]
a       = (G*MSun*MStar/Omega^2)^0.33333; % semimajor axis [m]
Fat1AU  = LStar*1366;                     % flux from GJ1132 at 1 AU [W/m2]

% for GJ 1132b
Fstel       = Fat1AU*(AU/a)^2;                % flux from GJ1132 at GJ1132b's orbit [W/m2]
Albedo       = 0.75;                           % planet albedo []

% % for Earth
% Fstel = 1366;
% Albedo = 0.3;

ASR   = (1-Albedo)*Fstel/4;    % absorbed stellar radiation [W/m2]
%Planet properties
MEarth = 5.97e24;%kg
REarth = 6371e3;%meters

% GJ 1132b
Mp = 1.62 * MEarth;
Rp = 1.16 * REarth;
Rc = 0.54 * Rp;
core_fraction = 0.262;

Mmantle = (1 - core_fraction) * Mp;
Teq = (ASR/sigma)^0.25;%K
gp = G * Mp / Rp^2;%m/s2
rho = Mmantle/(4/3*pi*(Rp^3-Rc^3)); %mantle density
Mmo = 4/3 * pi * rho* (Rp^3 - (Tr(2))^3); %mass of magma ocean
Wmo = (Tr(4) - Tr(3)); %mass of water in the magma ocean + atmosphere system)

% fun = Tr(5) - Teq;
fun = Tr(1) - 1000;
% fun = Tr(5) - 1420;
% fun = Tr(2)/Mmantle - 1e-9;
isterminal = 1;
direction = 0;

% [~,FH2O,~] = get_pressure2(Tr(7),Tr(2),Mmo,Mmantle,Rp,gp,Rc,Wmo);
% [~,meltfrac] = get_meltfrac(gp,Tr(1),Rp,Rc,Mmantle);
% 
% [qm,Db,uc,Ra,nu] = mantleheatflux(Tr(1),Tr(7),Tr(2),Rp,Rc,gp,rho,FH2O,meltfrac);
% 
% fun = Db - 1e3;
% isterminal = 1;
% direction = 0;
% 

end
