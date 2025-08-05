%get_heat_cap2.m
%calculate the heat capacity as a function of mantle temperature
%use intersection of adiabat and solid/liquidus
% function [Cp] = get_heat_cap2(Tnew,Told,Mmantle,g,Rp,Rc,Mmo)
function [Cp] = get_heat_cap3(Tm,Mmantle, g,Rp,Rc,Mmo,deltaMs)
    cp_kg = 1.2e3; %J/kg/K
    deltaH_kg = 4e5; %J/kg
    
% %     [~,melt_frac_new] = get_meltfrac(g,Tnew,Rp,Rc,Mmantle);
% %     
% %     [~,melt_frac_old] = get_meltfrac(g,Told,Rp,Rc,Mmantle);
%    
% %     At present, this if the volume-averaged melt fraction, but it should
% %     probably be the mass-averaged melt fraction. fix this later...
%     if Tnew == Told
%         dmelt_dT = 0.0;
%     else
%         dmelt_dT = (melt_frac_new - melt_frac_old) / (Tnew - Told);
%     end
%     
% %     Cp = (cp_kg + deltaH_kg * dmelt_dT) * Mmantle; % Joules/K for entire mantle
% %     Cp = (cp_kg + deltaH_kg * dmelt_dT) * Mmo; % Joules/K for entire mantle

    [~,meltfrac] = get_meltfrac(g,Tm,Rp,Rc,Mmantle);
    Cp = cp_kg * meltfrac * Mmo + deltaH_kg * deltaMs; % Joules/K for entire mantle    
end
    
    
