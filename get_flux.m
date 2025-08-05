%get_flux.m
function [flux] = get_flux(Tsurf,Teq,Psurf,Rp,g)
    sigma = 5.67e-8;    % Stefan-Boltzmann constant [W/m2/K4]
    k0 = 0.01; %absorption coefficient, water, m2/kg
    p0 = 1.01325e5; % reference pressure, Pa
    
    Matm = 4 * pi * Psurf * Rp^2 / g;
    tau = (3*Matm) / (8 * pi * Rp^2) * sqrt((k0*g)/(3*p0));
    emissivity = 2 / (tau + 2);
    
     flux = emissivity * sigma * (Tsurf^4 - Teq^4);
%      flux = sigma * (Tsurf^4 - Teq^4);

end