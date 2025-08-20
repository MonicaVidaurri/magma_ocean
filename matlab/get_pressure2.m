% get_pressure2.m
function [PH2O,FH2Ol,kH2O] = get_pressure2(Tm,rs,Mmo,Mmantle,Rp,g,Rc,MH2O)
% Get atmospheric pressure of water vapor in equilibium with atmosphere.
% Currently using equilibrium crystallization. Should update for fractional
% crystallization, and H loss (O production).

%partition coefficients for water between perovskite and melt 
rho = Mmantle/(4/3*pi*(Rp^3-Rc^3));

% solidus and liquidus temperatures in Kelvin
Tliquidus = 1973;
Tsolidus = 1373;



if Tm > Tsolidus
    
    baseP = rho * g * (Rp - rs)/1e9;
    [~,meltfrac] = get_meltfrac(g,Tm,Rp,Rc,Mmantle);
    Mliquid= meltfrac * Mmo; %not all of the "magma ocean" is liquid
    Msolid = (1-meltfrac) * Mmo; %solid 
    
    kH2O = 0.01;
   
    %solve for XH2O in the magma ocean using Newton's Method
    i = 1;
    FH2O_0 = MH2O/Mmantle;
    while i <= 100
        FH2O_new = FH2O_0 - massbalance(FH2O_0)/derivative(FH2O_0);
        if abs(FH2O_new - FH2O_0) < 1e-12
            FH2Ol = FH2O_new;
            break
        end
        i = i+1;
        FH2O_0 = FH2O_new;
    end

    FH2Os = FH2Ol*kH2O;
    PH2O = (FH2Ol/3.44e-8).^(1/0.74);% in Pa
    
else
   FH2Ol = 0.0;
   PH2O = MH2O * g / (4 * pi * Rp^2);
   kH2O = 0.0;

end

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fun = massbalance(x)%how are these different????
    fun = MH2O - x*(kH2O*Msolid + Mliquid) - (4 * pi * Rp^2 / g) .*...
        (x/3.44e-8).^(1/0.74);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function deriv = derivative(x)
    deriv = - (kH2O*Msolid + Mliquid) - (1/0.74)*(1/3.44e-8)^(1/0.74)*...
        (4 * pi * Rp^2 / g) .*x.^((1/0.74)-1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


end


