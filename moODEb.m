%this function will calculate the time derivatives of the mantle
%temperature T and radius of solidification r
% function [dTr_dt] = moODEb(t,Tr,Rp,Rc,Mmantle,Teq,rho,g,Ts,Ps,OLR,ASR,t_flux,phi,Xi,FeOt)
function [dTr_dt] = moODEb(t,Tr,Rp,Rc,Mmantle,Teq,rho,g,Ts,Ps,OLR,ASR,t_flux,L_GJ,Xi,FeOt,Temp_K,P_Pa,tsat,a,Mp,LStar)

%constants
sigma = 5.67e-8; %Boltzmann constant (W/m2/K4)
% Tsola = 26.53e-9; % solidus parameter (K/Pa)  Tsolidus = Tsola * P(Gpa) + Tsolb
% Tsolb = 1373; % solidus parameter (K)
dHf = 4e5; % heat of fusion of silicate (J/kg)
alpha = 2e-5;%thermal expansion coefficient (K^-1)
cp = 1.2e3; %heat capacity (J/kg/K)
muO = 15.9994e-3; %atomic weight of O (kg/mole)
muH2O = 18.015e-3; %molecular weight of H2O (kg/mole)
muFeO1_5 = 159.689e-3 / 2;        %molecular weight of FeO1.5 (kg/mole)
muFeO = 71.845e-3;                %molecular weight of FeO (kg/mole)
NA = 6.022e23;                      %Avogadros' number (molecules/mole)
muH = 1.008e-3;                     %atomic weight of H (kg/mole)
cpH2O = 2e3; %J/kg/K
% mid-ocean ridge length = 1.5x planet circumference (m)
Lridge = 2 * pi() * Rp * 1.5;
%%%re-compute in magma_ocean_evolution.m, once you get derivative result,
%%%plug back in to main file.

%Tr = [mantle temperature, radius of solidification,
%mass of water in solid mantle, total mass of water,mass of O2 in
%atmosphere, mole fraction FeO in magma ocea]
Tm = Tr(1); %surface temperature
rs = Tr(2); %radius of solidification
Mmo = 4/3 * pi * rho* (Rp^3 - (Tr(2))^3); %mass of magma ocean
if rs > Rp;
    rs = Rp;
    Mmo = 0;
end

Wmo = Tr(4); %mass of water in the magma ocean + atmosphere system)
if Wmo > 0;
    FH2O = Wmo / Mmo; %mass fraction of water in the magma ocean
else
    Wmo = 0.0;
    FH2O = 0.0;
end
MO_sol = Tr(6); % mass of free O in solid mantle (from O.5 in FeO1.5, includes original planetary abundance)
MO_mo = Tr(5); %mass of free O in magma ocean + atmosphere system
Tsurf = Tr(7);
FH2Os = Tr(3)/Mmantle;
dTr_dt = zeros(7,1);
% Tr
if imag(Tr)
    pause
end

if (Rp - rs) > 405e9 / 77.89/rho/g
    Tsola = 26.53e-9;
    Tsolb = 1825;
else
    Tsola = 104.42e-9;
    Tsolb = 1420;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dr_dt = B * dTm_dt where B is given below, assuming linear Taylor
%expansion of the adiabatic gradient and a linear solidus relationship
B = (cp *(Tsolb*alpha - Tsola*rho*cp)) / (g*(Tsola * rho * cp - alpha * Tm)^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%heat capacity is either: cp' = cp*Mmo or cp' = (cp + (1-meltfrc)*dHf)*Mmo;
%the second version may overcount heat of fusion, since it is also included
%in the ODE
    [~,meltfrac] = get_meltfrac(g,Tm,Rp,Rc,Mmantle);
% if Tsurf > 1373
if meltfrac >= 0.4;
    heatcap = cp* 4/3 * rho* pi * (Rp^3 - rs^3);
    % heatcap = (cp + dHf * (1-meltfrac) ) * 4/3 * pi * rho * (Rp^3 - rs^3);
else
    heatcap = cp * Mmantle;
%     meltfrac = 0.0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%find the atmospheric pressure (Pa)
if Wmo > 0.0;
    [Patm,FH2O,kH2O] = get_pressure2(Tm,rs,Mmo,Mmantle,Rp,g,Rc,Wmo);
else
    Patm = 0.0;
    FH2O = 0.0;
    kH2O = 0.0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Find the distribution of O between atmosphere and magma ocean
% if Tsurf > 1373
% if meltfrac >= 0.4    
if MO_mo > 0.0
    [PO2,FFeO1_5,~,~] = get_massbalance4(Tm,Patm,Mmo,MO_mo,Xi,FeOt,g,Rp);
    if PO2 < 0.0;
        PO2 = MO_mo * g / (4 * pi * Rp^2);
        FFeO1_5 = 0.0;
    end
else
    PO2 = 0.0;
    FFeO1_5 = 0.0;
end
% else
%     PO2 = MO_mo * g / (4 * pi * Rp^2);
%     FFeO1_5 = 0.0;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Atmospheric Heat Flux from LBL (W/m2)
% if (Patm > 1 && Tsurf > 400) && (Patm < 1000e5 && Tsurf < 4000)
%     flux = interp2(Ts,Ps,OLR,Tsurf,Patm)-ASR;
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
flux = get_flux(Tsurf,Teq,Patm,Rp,g);
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
    Q = 6.15e13;

elseif 1.01e9 < t < 2.75e9
    % For now ignore mantle pressure
    mantle_pressure = 0.0;

    [tidal_heating] = get_tidalheating(t, Tm, mantle_pressure, mantle_volume);
% % 
% % elseif 1.01e9 < t < 2.75e9
% %     Q = 0.0039*t^6 - 33.54*t^5 + 102768*t^4 - 1e8*t^3 - 2e10*t^2 + 1e14*t - 8e16 
% 
elseif t > 2.75e9
    Q = 6.15e15;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%find mantle heat flux
[q_mantle,Db,uc,Ra,nu] = mantleheatflux(Tm,Tsurf,rs,Rp,Rc,g,rho,FH2Os,meltfrac);

%spreading rate (m^2/s)
S =  2 * Lridge * uc; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Differential Equations
% The change in temperature depends on the rate of change of the
% solidification radius, but once the radius reaches the surface (i.e., no
% more mamga ocean, there should no longer be any dependence on it. In
% future models, this is where I should switch to the mantle degassing
% model. 


%     before magma ocean soldification
% if rs/Rp < 1.0
% if Tsurf > 1373    
% if Db < 1e3;

    %dTr_dt(1) = mantle potential temperature (K/yr)
    if rs < Rp
        dTr_dt(1) = 3.15569e7 * (-4*pi * Rp^2 * q_mantle + Q) / (cp * 4/3*...
            pi*rho*(Rp^3 - rs^3) - (4*pi*rho * dHf * rs^2)*B);
    else
        dTr_dt(1) = 3.15569e7 * (-4*pi * Rp^2 * q_mantle + Q) / (cp * Mmantle);
    end
        
   
    %dTr_dt(2) = solidification radius
    if rs < Rp
        dTr_dt(2) = B * dTr_dt(1);
    else
        dTr_dt(2) = 0.0;
    end
    
    %dTr_dt(3) = abundance of water within the solid mantle
    if FH2O > 0 %Tr(3) > 0;
        dTr_dt(3) = -kH2O * FH2O * (-4 * pi * rho * rs^2 * dTr_dt(2));
    else
        dTr_dt(3) = 0;
    end
    
    % dTr_dt(4) = water abundance in magma ocean + atmosphere
    if Wmo > 0;
        dTr_dt(4) = -dTr_dt(3) - 3.15569e7*4*pi*Rp^2*phi_H * muH2O/2/muH;
    else
        dTr_dt(4) = 0;
    end

    %dTr_dt(6) = abundance of free oxygen in solid mantle
    dTr_dt(6) = FFeO1_5 * 4 * pi * rho * rs^2 * dTr_dt(2) * 0.5 * muO / ...
        muFeO1_5; %mass of free O in solid mantle

    % dTr_dt(5) = free oxygen abudnance in m.o. + atmosphere
%     if Patm * 2 * muH / muH2O > PO2 && PO2 > 0
%     if PO2 > 0
    if Wmo > 0 && PO2 > 0
        dTr_dt(5) = 4 * pi * Rp^2 * 3.15569e7 *(phi_H * muO/2/muH -  ...
            phi_O) - dTr_dt(6); %(gain of O from loss of H in water) + (atmospheric loss) + (loss to solid)
%         dTr_dt(5) = 4 * pi * Rp^2 * 3.15569e7 *(phi_H * muO/2/muH) -  ...
%              - dTr_dt(6); %(gain of O from loss of H in water) + (atmospheric loss) + (loss to solid)
    elseif Wmo > 0 && PO2 <= 0
        dTr_dt(5) = 4 * pi * Rp^2 * 3.15569e7 *(phi_H * muO/2/muH) - dTr_dt(6); %(gain of O from loss of H in water) + (loss to solid)
    elseif Wmo <= 0 && PO2 > 0
%         dTr_dt(5) = - 4 * pi * Rp^2 * 3.15569e7 *phi_O - dTr_dt(6); %(loss of atmospheric O) + (loss to solid)
        dTr_dt(5) = - dTr_dt(6); %(loss of atmospheric O) + (loss to solid)
    else
        dTr_dt(5) = 0;
    end

    %dTr_dt(7) =  surface temperature
    dTr_dt(7) = 3.15569e7 * 4*pi * Rp^2 * ((-flux + q_mantle)/(cpH2O * ...
        Patm * 4* pi * Rp^2/g + cp * 4/3 * pi * 3e3 * (Rp^3 - Db^3))); 
    

end
