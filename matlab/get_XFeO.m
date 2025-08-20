function [fO2,XFeO] = get_XFeO(Tm,P,Mmo,Rp,g,Xi,MO2_mo)

% %mole fractions of oxides
%     X(1) = XSiO2
%     X(2) = XTiO2
%     X(3) = XAl2O3
%     X(4) = XMgO
%     X(5) = XCaO
%     X(6) = XNa2O
%     X(7) = XK2O
%     X(8) = XP2O5
%     X(9) = XFeOt
%     X(10) = XFeO1_5
%     X(11) = XFeO
%     X(12) = XFe2O3
% 

options = optimoptions('fsolve','Display','off','TolX',1e-10);

PO2 = MO2_mo * g / (4 * pi * Rp^2);
myfun = @(x)get_fun(Tm,P,Xi,MO2_mo,Rp,g,Mmo,x);
x0 = [PO2*0.9; Xi(11)];
sol = fsolve(myfun,x0,options);

while sol(1) < 0 || ~isreal(sol(1))
%     PO2 = MO2_mo * g / (4 * pi * Rp^2);
    x0 = [x0(1)*0.8; 0.8*x0(2)];
    sol = fsolve(myfun,x0,options);
end

fO2 = sol(1);
XFeO = sol(2);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    function fun = get_fun(T,P,Xi,MO2_mo,Rp,g,Mmo,guess)
%         guess(1) = fO2
%         guess(2) = XFeO
        muO2 = 15.9994e-3*2; %molecular weight of O2, kg

        fun(1,1) = log(guess(1)) - (1/0.196 *(log((Xi(9)-guess(2))/guess(2)) - ...
            1.1492e4./T + 6.675 + 2.243 * Xi(3) + 1.828 * Xi(9) - 3.201 * Xi(5) - ...
            5.854 * Xi(6) - 6.215 * Xi(7) + 3.36 *(1 - 1673./ T - log(T./1673)) +...
            7.01e-7 .* P./T + 1.54e-10 * (T - 1673).*P./T - 3.85e-17 * P.^2 ./ T));
        
        fun(2,1) = MO2_mo -4 *pi * Rp^2 * guess(1) / g - (Xi(9) - guess(2) -...
            Xi(12)) * 0.25 * muO2 * Mmo;
    
    
    end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 


end

