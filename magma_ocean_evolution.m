%magma_ocean_evolution.m
%Magma Ocean Model for GJ 1132b
%including separate evolution equation for radius of solidification so that
%I can actually use an ode solver
clear;clc;

%constants
G = 6.67e-11;%m3/kg/s2
sigma = 5.67e-8; %W/m2/K4
load('OLRdatab.mat')
[Ts,Ps] = meshgrid(Temp_K,P_Pa);
% load M-star luminosity track data from Baraffe+2015
% linearly interpolated between 0.1 M_Sun and 0.2 M_Sun to GJ1132 mass
% data is L_GJ [L_Sun] and t [yr].
load('stellar_dataTrappist1.mat'); t_flux = t; clear t;
% model = 1; %XUV Flux A
model = 2; %XUV Flux B
tsat = 1e9;


% Star properties
MSun    = 1.989e30;                       % sun mass [kg]
LSun    = 3.846e26;                       % sun luminosity [W]
AU      = 149597871*1e3;                  % earth semimajor axis [m]
% % sun
% MStar   = 1.0;                     
% LStar   = 1.0;                       
% Tday    = 1.0;                  

% Omega   = 2*pi/(Tday*86400);              % orbital frequency [rad/s]
% a       = (G*MSun*MStar/Omega^2)^0.33333; % semimajor axis [m]

%Trappist - 1
MStar   = 0.0898;                          % mass of star
LStar   = 0.000553;                        % luminosity of star

%Proxima Centauri Faria et al. 2022
% MStar   = 0.1221;
% LStar   = 0.0016;

%TOI 700- Rodriguez et al. 2020
% MStar = 0.415
% LStar = 0.0232

%Planet orbital distance
% a = 1.58e-2*AU; %Trappist-1c
% a = 2.227e-2*AU; %Trappist-1d
a = 2.925e-2*AU; %Trappist-1e
% a = 3.849e-2*AU; %Trappist-1f
% a = 4.8e-2*AU;    %Proxima b
% a = 16.33e-2*AU   %TOI 700-d
% a = 1*AU           %erf

Fat1AU  = LStar*1366;                     % flux at 1 AU [W/m2]
Fstel       = Fat1AU*(AU/a)^2;                % flux from GJ1132 at GJ1132b's orbit [W/m2]
Albedo       = 0.25;                           % planet albedo []
ASR   = (1-Albedo)*Fstel/4;    % absorbed stellar radiation [W/m2]

%Planet properties
MEarth = 5.97e24;%kg
REarth = 6371e3;%meters
% % erf
% Rc = 0.19 * REarth
% core_fraction = 0.16
% Mp = 1 * MEarth
% Rp = 1 * REarth

% % Trappist-1c Agol et al. 2021
% Mp = 1.308 * MEarth; 
% Rp = 1.097 * REarth;
% Rc = 0.518 * Rp;
% core_fraction = 0.266;

% % Trappist-1d Agol et al. 2021
% Mp = 0.388 * MEarth; 
% Rp = 0.788 * REarth;
% Rc = 0.426 * Rp;
% core_fraction = 0.177;

% Trappist-1e Agol et al. 2021
Mp = 0.692 * MEarth; 
Rp = 0.92 * REarth;
Rc = 0.467 * Rp;
core_fraction = 0.236;

% % Trappist-1f Agol et al. 2021
% Mp = 1.039 * MEarth; 
% Rp = 1.045 * REarth;
% Rc = 0.429 * Rp;
% core_fraction = 0.192;

% % Proxima Centauri b Faria et al. 2022
% Mp = 1.07 * MEarth;
% Rp = 1.03 * REarth;
% Rc = 0.429 * Rp;
% core_fraction = 0.2;

% % TOI 700-d Rodriguez et al. 2020
% Mp = 1.9 * MEarth;
% Rp = 1.144 * REarth;
% Rc = 0.5 * Rp;
% core_fraction = 0.2;

Mmantle = (1 - core_fraction) * Mp;
Earth_ocean = 1.39e21;%kg
Teq = (ASR/sigma)^0.25;%K
gp = G * Mp / Rp^2;%m/s2


%SET WATER ABUNDANCE HERE
MH2O = 2 * Earth_ocean;
FH2O = MH2O/Mmantle;
FeOt = 0.08;
Fe3_Fet = 1e-12;


%solidus temperature parameters
Tsol1 = 26.53e-9; %K/Pa
Tsol2 = 1373; %k
Cp = 1.2e3; %heat capacity (J/kg/K)
alpha = 2e-5;%thermal expansion (1/K)
rho = Mmantle/(4/3*pi*(Rp^3-Rc^3)); %mantle density
Xi = get_comp(1); %import the mole fraction of the oxides, 1 = BSE, 2 = BSMars
muO = 15.9994e-3;                 % atomic weight of O in (kg/mole)
muFeO1_5 = 159.689e-3 / 2;        %molecular weight of FeO1.5 (kg/mole)
muFeO = 71.845e-3;                %molecular weight of FeO (kg/mole)


%initial conditions for a full mantle magma ocean
%use 3700 for 1c
%use 2500 for 1d,
%use 3000 for 1e
%use 3500 for 1f
Tr0(1) = 3000;                                                      %intial potential temperature temperature
Tr0(2) = Rp - (Tr0(1) - Tsol2)*Cp/(Tsol1*rho*gp*Cp - alpha*gp*Tr0(1)); %initial depth to base of magma ocean
if Tr0(2) < Rc
    Tr0(2) = Rc;
end
Mmo0 = (4/3 * pi * rho * (Rp^3 - Tr0(2)^3));                      %initial mass of magma ocean
Tr0(3) = (MH2O - FH2O * Mmo0);                          %intial abundance H2O in solid phase
Tr0(4) = FH2O * Mmo0;                                   %total initial water abundance in m.o. + atm.
Tr0(5) = FeOt * Fe3_Fet * Mmo0 * (muO / 2 / muFeO1_5);  %initial mass of O in FeO1.5 in m.o. + atm
Tr0(6) = FeOt * Fe3_Fet * (Mmantle - Mmo0)* (muO / 2 / muFeO1_5);  % mass of O locked up in solid FeO1.5
Tr0(7) = Tr0(1)-1; %initial surface temperature is equal to the intial potential temperature


%use event location to stop the model once Tm equals the solidus.
options = odeset('Events',@mo_event,'NonNegative',[1 2 3 4 5 6 7]);
% options = odeset('NonNegative',[1 2 3 4 5 6 7],);
options2 = odeset('NonNegative',[1 2 3 4 5],'Events',@mo_event2);
% options2 = odeset('NonNegative',[1 2 3 4 5],'Events',@mo_event3);
options3 = odeset('NonNegative',[1 2 3 4],'Events',@mo_event3);
% 
%     options = odeset('Events',@mo_event,'NonNegative',[1 2 3 4 5 6 7],'RelTol',1e-8,'AbsTol',1e-6*[1e-15 1e-12 1e0 1e0 1e-6 1e-6 1e-15]);
%     % options = odeset('NonNegative',[1 2 3 4 5 6 7]);
%     options2 = odeset('NonNegative',[1 2 3 4 5],'Events',@mo_event3,'RelTol',1e-8,'AbsTol',1e-6*[1e-15 1e0 1e0 1e0 1e-15]);

[t,Tr,TE,RE,YE] = ode23t(@(t,Tr)moODEb(t,Tr,Rp,Rc,Mmantle,Teq,rho,gp,Ts,Ps,...
    OLR,ASR,t_flux,L_GJ,Xi,FeOt,Temp_K,P_Pa,tsat,a,Mp,LStar),[1 7.6e9],Tr0,options);

for i = 1:length(t)
    Mmo(i) = 4/3 * pi * rho* (Rp^3 - Tr(i,2)^3);
    
    Wmo(i) = Tr(i,4);
    if Wmo(i) < 0
        Wmo(i) = 0.0;
    end
    
    [~,meltfrac(i)] = get_meltfrac(gp,Tr(i,1),Rp,Rc,Mmantle);
    
    if Wmo(i) > 0
        [Patm(i),FH2O(i),kH2O(i)] = get_pressure2(Tr(i,1),Tr(i,2),Mmo(i),...
            Mmantle,Rp,gp,Rc,Wmo(i));
    else
        Patm(i) = 0.0;
        FH2O(i) = 0.0;
        kH2O(i) = 0.0;
    end
    
%     if (Patm(i) > 1 && Tr(i,7) > 400) && (Patm(i) < 1000e5 && Tr(i,7) < 4000)
%         flux(i) = interp2(Ts,Ps,OLR,Tr(i,7),Patm(i))-ASR;
%     else
        flux = get_flux(Tr(i,7),Teq,Patm(i),Rp,gp);   
%         [logPs,Ts] = ndgrid(log10(P_Pa),Temp_K);
%         F =griddedInterpolant(logPs,Ts,log10(OLR),'cubic');
%         F2 = griddedInterpolant(logPs,Ts,log10(OLR));
%         newOLR = (10^F(log10(Patm(i)),Tr(i,7)) + 10^F2(log10(Patm(i)),Tr(i,7)))/2;
%         if newOLR < min(min(OLR));
%             newOLR = min(min(OLR));
%         end
%         flux(i) = newOLR - ASR;
%     end    
    
    Hres(i,1) = FH2O(i) * meltfrac(i)*Mmo(i);
    Hres(i,2) = 4*pi*Rp^2/gp * Patm(i);
    Hres(i,3) = Tr(i,3) + kH2O(i) * FH2O(i)*(1-meltfrac(i))*Mmo(i);
    
    MO_mo(i) = Tr(i,5);
if MO_mo(i) > 0.0    
    [PO2(i),FFeO1_5(i),mark(i),nFeO1_5(i)] = get_massbalance4(Tr(i,1),Patm(i),Mmo(i),...
        MO_mo(i),Xi,FeOt,gp,Rp);
    if PO2(i) < 0.0;
        PO2(i) = MO_mo(i) * gp / (4 * pi * Rp^2);
        FFeO1_5(i) = 0.0;
        nFeO1_5(i) = 0.0;
    end
else
    PO2(i) = 0.0;
    FFeO1_5(i) = 0.0;
    mark(i) = 0.0;
    nFeO1_5(i) = 0.0;
end
nFeOt(i) = FeOt * Mmo(i)/muFeO;           %moles of total Fe in magma ocean

[fo2(i)] = get_fO2(Tr(i,1),Patm(i),[Xi(1:10) nFeOt(i)-nFeO1_5(i) nFeO1_5(i)]);
[phi_H(i),phi_O(i)] = get_loss(t_flux,L_GJ,t(i),PO2(i),Patm(i),tsat,a,Mp,Rp,LStar);

    
end

    H2Olost = MH2O - (Tr(:,3) + Tr(:,4));
    Ogained = H2Olost*muO/18.015e-3;
    totalO = Tr0(5) + Ogained; %neglects loss of O from atmosphere

%%%%%%%%%%% begin finding mantle heat flux shit for T1 %%%%%%%%%%%%%%
% m_heatflux = Tr(:,7)*mantleheatflux(Tr0(1),Tr0(7),Tr(2),Rp,Rc,gp,rho,FH2O,meltfrac)

%%joe's results are broken down into 3 segments

% 
% elseif 1.01e9 < t < 2.75e9
%     Q = 0.0039*t^6 - 33.54*t^5 + 102768*t^4 - 1e8*t^3 - 2e10*t^2 + 1e14*t - 8e16 
% 
% elseif t > 1.01e9
%     Qw = 6.15e15
% end
% % Q = Tr(:,7)*Qw


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tr0_2(1) = Tr(end,1); %mantle temperature
Tr0_2(2) = Tr(end,3) + Tr(end,4) - Patm(end)*4*pi*Rp^2/gp; %water in mantle (include any water that may still have been in melt)
% Tr0_2(3) = Tr(end,4); %total water
Tr0_2(3) = Patm(end)*4*pi*Rp^2/gp; %water in the atmosphere
Tr0_2(4) = PO2(end)*4*pi*Rp^2/gp; %total O in atmosphere
Tr0_2(5) = Tr(end,7); %surface temperature

[t2,Tr2,TE,RE,YE] = ode23t(@(t2,Tr2)moODEb2(t2,Tr2,Rp,Rc,Mmantle,Teq,rho,gp,Ts,Ps,...
    OLR,ASR,t_flux,L_GJ,Xi,FeOt,Temp_K,P_Pa,tsat,a,Mp,LStar),[t(end) 7.6e9],Tr0_2,options2);

PO2_2 = Tr2(:,4)*gp/4/pi/Rp^2; %O2 pressure in Pa
Patm_2 = Tr2(:,3)*gp/4/pi/Rp^2;%H2O pressure in Pa

%%%%%%%%%%% begin finding mantle heat flux shit for T1 %%%%%%%%%%%%%%
% m_heatflux_2 = Tr2(:,5)*mantleheatflux(Tr0_2(1),Tr0_2(5),Tr2(2),Rp,Rc,gp,rho,FH2O,meltfrac)

% %%joe's results are broken down into 3 segments
% if t <= 1.01e9
%     Qw_2 = 6.15e13*(t<=1.01e9)
% % 
% % elseif 1.01e9 < t < 2.75e9
% %     Q = 0.0039*t^6 - 33.54*t^5 + 102768*t^4 - 1e8*t^3 - 2e10*t^2 + 1e14*t - 8e16 
% 
% elseif t > 1.01e9
%     Qw_2 = 6.15e15
% end
% % Q_2 = Tr(:,6)*Qw_2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tr0_3(1) = Tr2(end,1); %mantle temperature
Tr0_3(2) = Tr2(end,3); %water in the atmosphere
Tr0_3(3) = Tr2(end,4); %total O in atmosphere
Tr0_3(4) = Tr2(end,5); %surface temperature
FH2O3 = Tr2(end,2)/Mmantle;

[t3,Tr3,TE,RE,YE] = ode23t(@(t3,Tr3)moODE3(t3,Tr3,Rp,Rc,Mmantle,Teq,rho,gp,Ts,Ps,...
    OLR,ASR,t_flux,L_GJ,Xi,FeOt,Temp_K,P_Pa,tsat,FH2O3,a,Mp,LStar),[t2(end) 7.6e9],Tr0_3,options3);

PO2_3 = Tr3(:,3)*gp/4/pi/Rp^2;%pressure in Pa
Patm_3 = Tr3(:,2)*gp/4/pi/Rp^2;%H2O pressure in Pa

%%%%%%%%%%% begin finding mantle heat flux shit for T1 %%%%%%%%%%%%%%
% m_heatflux_3 = Tr3(:,4)*mantleheatflux(Tr0_3(1),Tr0_3(4),Tr3(2),Rp,Rc,gp,rho,FH2O,meltfrac)

%%joe's results are broken down into 3 segments
% if t <= 1.01e9
%     Qw_3 = 6.15e13*(t<=1.01e9)
%     disp('less')
% % 
% % elseif 1.01e9 < t < 2.75e9
% %     Q = 0.0039*t^6 - 33.54*t^5 + 102768*t^4 - 1e8*t^3 - 2e10*t^2 + 1e14*t - 8e16 
% 
% elseif t > 1.01e9
%     Qw_3 = 12e13
%     disp('hello')
% end
% % Q_3 = Tr(:,5)*Qw_3

% vis = viscosity(Tr0_3(1),Tr0_3(4),rho,FH2O,meltfrac)