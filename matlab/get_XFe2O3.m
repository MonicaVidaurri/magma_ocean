function [XFe2O3] = get_XFe2O3(T,P,X,fO2,x0)

% %mole fractions of oxides
% X(1): XAl2O3 
% X(2): XCaO
% X(3): XNa2O
% X(4): XK2O
% X(5): XFeOt
% X(6): XFeO1_5 

% XFeO = X(5) - XFeO1_5;
% XFe2O3 = XFeO1_5;
options = optimoptions('fsolve','Display','off');
lnfO2 = log(fO2);

myfun = @(x)get_fun(T,P,X,lnfO2,x);
% x0 = 0.02;
XFe2O3 = fsolve(myfun,x0,options);


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    function fun = get_fun(T,P,X,lnfO2,Fe2O3guess)

    fun = lnfO2 - (1/0.196 *(log(Fe2O3guess) - ...
        1.1492e4./T + 6.675 + 2.243 * X(3) + 1.828 * X(9) - 3.201 * X(5) - ...
        5.854 * X(6) - 6.215 * X(7) + 3.36 *(1 - 1673./ T - log(T./1673)) +...
        7.01e-7 .* P./T + 1.54e-10 * (T - 1673).*P./T - 3.85e-17 * P.^2 ./ T));
    
        end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 


end

