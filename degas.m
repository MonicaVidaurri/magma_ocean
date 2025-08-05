function [Dmelt,rmor,avgfmelt,avgXmelt] = degas(Tm,Db,qm,FH2O,Rp,g,Tsurf)
% function [Tprofile, Tsolidus,Tsol_wet,Tliquidus,Tliq_wet,press,z,Dmelt] = degassing3(temp,Db,qm,f_water,Rp,g,avgfact)

%This function calculates the degassing rate. To do this, we must first
%calculate the melt fraction in the melt zone, the weight fraction of water
%in the melt and the thickness of the melt zone. 

%bulk distribution of water between silicate and melt
D_H2O = 0.01;

%density
rho_m = 3.3e3;
beta = (3/2);
% parameters to calculate offset from solidus due to water
K = 43; %degrees C / wt%^gamma
gamma = (3/4);
alpha = 2e-5;
cp=1.2e3;
chi_d = 1;
km = 4.2;
% Rp = 6271e3;

%pressure (GPa) from surface to an arbitrary depth. 300 km is what Sandu et
%al. used. May need to modify this in future.
z = (0:1e3:300e3)'; %meters
y = zeros(length(z),1);
Xmelt = zeros(length(z),1);
Tprofile = zeros(length(z),1);

press = rho_m * g * z/1e9; %in GPa
% Tsolidus = 26.53 * press + 1373;
% Tliquidus = 1973 + 26.53 * press;
Tsolidus = min(104.42*press + 1420, 26.53*press+1825);
Tliquidus = Tsolidus + 600;

%calculate temperature profile through the boundary layer and upper mantle
Tprofile(1) = Tsurf;%K
%mantle potential temperature
Tp = Tm;
for i = 2:length(z)
    if z(i) < Db
        %boundary layer thermal conductive temperature profile
        Tprofile(i) = Tprofile(1) + z(i) * qm/km; %K
    else
%         mantle adiabat
        Tprofile(i) = Tp + Tp *(alpha*g*(z(i))/cp);
    end
end

y(1)=0;
Xmelt(1) = 0;
options = optimset('Display','off');

%try finding Fmelt using fzero or fsolve
for i = 2:length(z)
    
    if Tprofile(i) >= Tliquidus(i)
        
        y(i) = 1.0;
                
    elseif Tprofile(i) < Tliquidus(i) && Tprofile(i) >= Tsolidus(i)
        
        y(i) = (Tprofile(i) - Tsolidus(i))/(Tliquidus(i) - Tsolidus(i));
        
        if y(i) <= 0.0;
            y(i) = 0.0;
        elseif y(i) > 1.0;
            y(i) = 1.0;
        end
        
    else
        
        y(i) = 0;
        
    end
    
    if y(i) > 0e0;
        
        Xmelt(i) = (FH2O/(D_H2O + y(i)*(1-D_H2O)));
        
    else
        
        Xmelt(i) = 0e0;
        
    end
    
end

%calculate the thickness of the melt layer
firstz = find(y,1,'first');
secondz = 1;
thirdz = 1;
fourthz = 1;
for i = firstz:length(z)
    if y(i) > 0.0e0;
        secondz = int16(i);
    end
    if i == length(z);
        break
    end
    if y(i+1) <= 0.0e0;
        break
    end
end
n = secondz + 1;
if sum(y(n:end)) > 0;
    y2 = y(n:end);
    thirdz = find(y2,1,'first')+n-1;
    fourthz = length(z);
end

% last = find(y,1,'last');

if isempty(firstz)
    firstz = int16(1);
end

Dmelt = z(secondz)-z(firstz) + z(fourthz) - z(thirdz);

r = Rp - z;

%find average water and melt fraction over the melt layer thickness
if Dmelt > 0e0;
    if fourthz > 1
        product1 = trapz(r(firstz:fourthz),y(firstz:fourthz).*r(firstz:fourthz).^2);
        product2 = trapz(r(firstz:fourthz),Xmelt(firstz:fourthz).*r(firstz:fourthz).^2);
        avgfmelt = 3* product1/(r(secondz)^3 - r(firstz)^3 + r(fourthz)^3 - r(thirdz)^3);
        avgXmelt = 3* product2/(r(secondz)^3 - r(firstz)^3 + r(fourthz)^3 - r(thirdz)^3);
    else
        product1 = trapz(r(firstz:secondz),y(firstz:secondz).*r(firstz:secondz).^2);
        product2 = trapz(r(firstz:secondz),Xmelt(firstz:secondz).*r(firstz:secondz).^2);
        avgfmelt = 3* product1/(r(secondz)^3 - r(firstz)^3);
        avgXmelt = 3* product2/(r(secondz)^3 - r(firstz)^3);
    end
%     rmor = avgfmelt * avgXmelt * Dmelt * chi_d * rho_m;
    rmor = avgfmelt * avgXmelt*rho_m;
else
    avgfmelt = 0e0;
    avgXmelt = 0e0;
    rmor = 0e0;
end


end





