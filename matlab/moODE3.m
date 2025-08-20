%this function will calculate the time derivatives of the mantle
%temperature T and radius of solidification r
% function [dTr_dt] = moODEb(t,Tr,Rp,Rc,Mmantle,Teq,rho,g,Ts,Ps,OLR,ASR,t_flux,phi,Xi,FeOt)
function [dTr_dt] = moODE3(t,Tr,Rp,Rc,Mmantle,Teq,rho,g,Ts,Ps,OLR,ASR,t_flux,L_GJ,Xi,FeOt,Temp_K,P_Pa,tsat,FH2O,a,Mp,LStar)

%constantts
sigma = 5.67e-8; %Boltzmann constant (W/m2/K4)
Tsola = 26.53e-9; % solidus parameter (K/Pa)  Tsolidus = Tsola * P(Gpa) + Tsolb
Tsolb = 1373; % solidus parameter (K)
dHf = 4e5; % heat of fusion of silicate (J/kg)
alpha = 2e-5;%thermal expansion coefficient (K^-1)
cp = 1.2e3; %heat capacity (J/kg/K)
muO = 15.9994e-3; %atomic weight of O (kg/mole)
muH2O = 18.015e-3; %molecular weight of H2O (kg/mole)
NA = 6.022e23;                      %Avogadros' number (molecules/mole)
muH = 1.008e-3;                     %atomic weight of H (kg/mole)
cpH2O = 2e3; %J/kg/K
% mid-ocean ridge length = 1.5x planet circumference (m)
Lridge = 2 * pi() * Rp;


%Tr = [mantle temperature, radius of solidification,
%mass of water in solid mantle, total mass of water,mass of O2 in
%atmosphere, mole fraction FeO in magma ocea]
Tm = Tr(1); %surface temperature
Watm = Tr(2); % mass of water in the atmosphere
if Watm < 0
    Watm = 0.0;
end

MO_tot = Tr(3); % mass of free O in atmosphere
if MO_tot < 0
    MO_tot = 0;
end
Tsurf = Tr(4);
dTr_dt = zeros(4,1);
t

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%heat capacity is either: cp' = cp*Mmo or cp' = (cp + (1-meltfrc)*dHf)*Mmo;
%the second version may overcount heat of fusion, since it is also included
%in the ODE
    heatcap = cp * Mmantle;
    if Tm > 1420
        [~,meltfrac] = get_meltfracb(g,Tm,Rp,Rc,Mmantle);
    else
        meltfrac = 0.0;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%find the atmospheric pressure (Pa)
    if Tr(4) > 647
        Patm = Watm * g / (4 * pi * Rp^2);
    else
        Patm = 10^(6.079-2261.10/Tr(4))*1e5;
        Matm = Patm * (4 * pi * Rp^2) / g;
        if Matm > Watm
            Patm = Watm * g / (4 * pi * Rp^2);
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Find the distribution of O between atmosphere and magma ocean
    PO2 = MO_tot * g / (4 * pi * Rp^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Atmospheric Heat Flux from LBL (W/m2)
% if Patm > 1 && Tsurf > 400 && Patm < 1000e5 && Tsurf < 4000
%     flux = interp2(Ts,Ps,OLR,Tsurf,Patm)-ASR;
% elseif Patm == 0
    flux = get_flux(Tsurf,Teq,Patm,Rp,g);
% elseif Patm > 1000e5 && Tsurf < 2200 %without this, at low temp and high P, the interpolation starts making flux go up!!!
%     newOLR = min(min(OLR));
%     flux = newOLR - ASR;
% else
% %     flux = get_flux(Tsurf,Teq,Patm,Rp,g);   
%     [logPs,Ts] = ndgrid(log10(P_Pa),Temp_K);
%     F =griddedInterpolant(logPs,Ts,log10(OLR),'cubic');
%     F2 = griddedInterpolant(logPs,Ts,log10(OLR));
%     newOLR = (10^F(log10(Patm),Tsurf) + 10^F2(log10(Patm),Tsurf))/2;
%     if newOLR < min(min(OLR));
%         newOLR = min(min(OLR));
%     end
%     flux = newOLR - ASR;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%find Atmospheric mass loss rate(kg/m2/s)
%Assuming for the time being that H loss is effectively H2O loss (removal
%of absorber)
[phi_H,phi_O] = get_loss(t_flux,L_GJ,t,PO2,Patm,tsat,a,Mp,Rp,LStar);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Radioactive Heat Production
% adding in radioactivity
% Radioactive heat constants (W/kg), from Schubert et al. (2001) Ch. 4
H_238U = 9.37e-5;
H_235U = 5.69e-4;
H_232Th = 2.69e-5;
H_40K = 2.79e-5;
%Radioisotope abundances, Schubert et al. (2001) Ch. 4
Uran = 21e-9; %[U] = 21 ppb
C_238U = 0.9927 * Uran;
C_235U = 0.0072 * Uran;
C_40K = 1.28 * Uran;
C_232Th = 4.01 * Uran;
%Decay constants (1/yr), from McNamara and van Keken (2000)
l_238U = 0.155e-9;
l_235U = 0.985e-9;
l_232Th = 0.0495e-9;
l_40K = 0.555e-9;

%heat production (W)
% Q = (C_238U * H_238U * exp(l_238U*(4.6e9 -t)) + C_235U*H_235U*exp(...
%     l_235U*(4.6e9 -t)) + C_232Th * H_232Th * exp(l_232Th * (4.6e9 -...
%     t)) + C_40K * H_40K * exp(l_40K * (4.6e9 -t)))*Mmantle;

%%joe's results are broken down into 3 segments
if t <= 1.01e9
    Q = 6.15e13

elseif 1.01e9 < t < 2.75e9
    [tidal_heating] = get_tidalheating(t, mantle_temp, mantle_pressure, mantle_volume)
% % 
% % elseif 1.01e9 < t < 2.75e9
% %     Q = 0.0039*t^6 - 33.54*t^5 + 102768*t^4 - 1e8*t^3 - 2e10*t^2 + 1e14*t - 8e16 
% 
elseif t > 2.75e9
    Q = 6.15e15
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%find mantle heat flux
[q_mantle,Db,uc,Ra,nu] = mantleheatflux(Tm,Tsurf,Rp,Rp,Rc,g,rho,FH2O,meltfrac);
if isnan(q_mantle)
    q_mantle
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Differential Equations
% The change in temperature depends on the rate of change of the
% solidification radius, but once the radius reaches the surface (i.e., no
% more mamga ocean, there should no longer be any dependence on it. In
% future models, this is where I should switch to the mantle degassing
% model. 

    dTr_dt(1) = 3.15569e7*(-4*pi * Rp^2 * q_mantle + Q) / (cp * Mmantle); %mantle potential temperature

%     dTr_dt(2) = (- 4 * pi * Rp^2 * phi_H*muH2O/2/muH) * 3.15569e7;%rate of loss of H2O from atmosphere
% 
%     dTr_dt(3) = (4 * pi * Rp^2 * phi_H * muO/2/muH - 4 * pi * Rp^2 *...
%         phi_O) * 3.15569e7; %gain of O from loss of atmospheric H
%     dTr_dt(3) = (4 * pi * Rp^2 * phi_H * muO/2/muH) * 3.15569e7; %gain of O from loss of atmospheric H
        
    if Tr(4) > 647 
        dTr_dt(2) = (- 4 * pi * Rp^2 * phi_H*muH2O/2/muH) * 3.15569e7;%rate of loss of H2O from atmosphere
    else 
        dTr_dt(2) = 0.0;
    end
    
    if Tr(4) > 647
        dTr_dt(3) = (4 * pi * Rp^2 * phi_H * muO/2/muH - 4 * pi * Rp^2 *...
            phi_O) * 3.15569e7; %gain of O from loss of atmospheric H
    else
        dTr_dt(3) = 0.0;
    end


    dTr_dt(4) = 4*pi * Rp^2 * 3.15569e7* ((-flux + q_mantle)/(cpH2O * Patm *...
        4* pi * Rp^2/g + cp * 4/3 * pi * 3e3 * (Rp^3 - Db^3))); 
%  if Tr(2) < 1e6
%      Tr(2);
%  end
%  
% if isnan(dTr_dt(4))
%     dTr_dt(4);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end
