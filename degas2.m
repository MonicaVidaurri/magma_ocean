function [Dmelt,rmor,meltfrac,avgXmelt] = degas2(Tm,Db,qm,FH2O,Rp,g,Tsurf)
% function [Tprofile, Tsolidus,Tsol_wet,Tliquidus,Tliq_wet,press,z,Dmelt] = degassing3(temp,Db,qm,f_water,Rp,g,avgfact)

%This function calculates the degassing rate. To do this, we must first
%calculate the melt fraction in the melt zone, the weight fraction of water
%in the melt and the thickness of the melt zone. 

%bulk distribution of water between silicate and melt
D_H2O = 0.01;

%density
rho_m = 3.3e3;
% parameters to calculate offset from solidus due to water
alpha = 2e-5;
cp=1.2e3;
chi_d = 1;
km = 4.2;

%pressure (GPa) from surface to an arbitrary depth. 300 km is what Sandu et
%al. used. May need to modify this in future.
z = (0:1e3:300e3)'; %meters
y = zeros(length(z),1);
Xmelt = zeros(length(z),1);
Tadiabat = zeros(length(z),1);

press = rho_m * g * z/1e9; %in GPa
% Tsolidus = 26.53 * press + 1373;
% Tliquidus = 1973 + 26.53 * press;
Tsolidus = min(104.42*press + 1420, 26.53*press+1825);
Tliquidus = Tsolidus + 600;

%calculate temperature profile through the boundary layer and upper mantle
Tadiabat(1) = Tsurf;%K
%mantle potential temperature
Tp = Tm;
for i = 2:length(z)
    if z(i) < Db
        %boundary layer thermal conductive temperature profile
        Tadiabat(i) = Tadiabat(1) + z(i) * qm/km; %K
    else
%         mantle adiabat
        Tadiabat(i) = Tp + Tp *(alpha*g*(z(i))/cp);
    end
end

Xmelt = zeros(1,length(z));
fraction = zeros(1,length(z));

%find melt fraction for each layer
for i = 1:length(z)
    if Tadiabat(i) > Tliquidus(i)
        fraction(i) = 1.0;
        Xmelt(i) = FH2O;
    elseif Tadiabat(i) > Tsolidus(i)
        fraction(i) = (Tadiabat(i) - Tsolidus(i)) / (Tliquidus(i) - Tsolidus(i));
        Xmelt(i) = (FH2O/(D_H2O + fraction(i)*(1-D_H2O)));
    else
        fraction(i) = 0e0;
        Xmelt(i) = 0.0;
    end
end

%find the pressure of the base of the melt
I = find(fraction>0);

%if the planet has cooled below the solidus, there is no melt. Set
%meltfraction and exchange pressure to 0
if length(I) <2
    meltfrac = 0.0e0;
    exchangeP = 0.0e0;
    avgXmelt = 0.0;
    Dmelt = 0.0;
else
    %find the base of the magma ocean
    if I(end) < length(fraction) 
        exchangeP = press(I(end)+1); %mantle is only partially molten
    else
        exchangeP = press(I(end)); %whole mantle is molten
    end

%     %find the volume-averaged melt fraction over the melt region
    r = (Rp - z(I(1):I(end)))';
    meltfrac = 3 * trapz(r(1:end),fraction(I(1):I(end)).*r(1:end).^2) / (r(end)^3 - r(1)^3);
    avgXmelt = 3 * trapz(r(1:end),Xmelt(I(1):I(end)).*r(1:end).^2) / (r(end)^3 - r(1)^3);
    
    %because of rounding issues, if whole mantle is molten, meltfrac > 1,
    %so I'm just arbitrarily setting it back to 1 here.
    if meltfrac > 1
        meltfrac = 1.0e0; 
    end
    Dmelt = z(I(end)) - z(I(1));

    
end

if Dmelt > 0
%     rmor = meltfrac * avgXmelt * Dmelt * chi_d * rho_m;
    rmor = meltfrac * avgXmelt*rho_m;
else
    rmor = 0e0;
end


end





