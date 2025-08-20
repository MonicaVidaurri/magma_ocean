function [exchangeP,meltfrac] = get_meltfrac(g,Tp,Rp,Rc,Mmantle)
%approximating the melt fraction throughout the magma ocean, and the
%depth/pressure of the base of the magma ocean by the intercept of the
%mantle adiabat with the solidus

%constants
% rho = 4.5e3; %density,kg/m3 (use silicate liquid EOS to get a better value here)
alpha  = 2e-5; %thermal expanson (1/K)
% rho = 5.01e3;
rho = Mmantle/(4/3*pi*(Rp^3-Rc^3));

cp = 1.2e3; %heat capacity
z = (0:5e3:(Rp-Rc)); %depth from surface, in meters
%pressure (GPa) from surface 
press = g*rho*z/1e9; %in GPa

% %linear approximation to solidus (K)
% Tsolidus = 26.53 * press + 1373;
% %liquidus approximation (K)
% Tliquidus = 1973 + 26.53 * press;
Tsolidus = min(104.42*press + 1420, 26.53*press+1825);
Tliquidus = Tsolidus + 600;


%mantle adiabat (K)
Tadiabat = Tp + Tp *(alpha*g*(z)/cp);

%find melt fraction for each layer
for i = 1:length(z)
    if Tadiabat(i) > Tliquidus(i)
        fraction(i) = 1.0;
    elseif Tadiabat(i) > Tsolidus(i)
        fraction(i) = (Tadiabat(i) - Tsolidus(i)) / (Tliquidus(i) - Tsolidus(i));
    else
        fraction(i) = 0e0;
    end
end

%find the pressure of the base of the magma ocean for the exchange reaction
I = find(fraction>0);

%if the planet has cooled below the solidus, there is no melt. Set
%meltfraction and exchange pressure to 0
if length(I) <2
    meltfrac = 0.0e0;
    exchangeP = 0.0e0;
else
    %find the base of the magma ocean
    if I(end) < length(fraction) 
        exchangeP = press(I(end)+1); %mantle is only partially molten
    else
        exchangeP = press(I(end)); %whole mantle is molten
    end

%     %find the volume-averaged melt fraction over the whole mantle
%     r = Rp - z;
%     find the volume-averaged melt fraction over only the magma ocean
    r = Rp - z(1:I(end));
    meltfrac = 3 * trapz(r(1:end),fraction(1:I(end)).*r(1:end).^2) / (r(end)^3 - r(1)^3);
    
    
    %because of rounding issues, if whole mantle is molten, meltfrac > 1,
    %so I'm just arbitrarily setting it back to 1 here.
    if meltfrac > 1
        meltfrac = 1.0e0; 
    end
    
    
end



end