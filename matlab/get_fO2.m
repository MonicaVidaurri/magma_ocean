function [fO2] = get_fO2(T,P,X)
% function [XFe2O3_FeO] = get_fO2(T,P,X,fO2)
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

lnfO2 = 1/0.196 *(log(X(12)/X(11)) - 1.1492e4./T + 6.675 + 2.243 * X(3) + ...
    1.828 * X(9) - 3.201 * X(5) - 5.854 * X(6) - 6.215 * X(7) + 3.36 *...
    (1 - 1673./ T - log(T./1673)) + 7.01e-7 .* P./T + 1.54e-10 * (T - 1673).*P./T - ...
    3.85e-17 * P.^2 ./ T);

fO2 = exp(lnfO2);

% logfO2 = log(fO2);
% 
% lnXFe2O3_FeO = 0.196 *logfO2 + 1.1492e4./T - 6.675 - 2.243 * X(3) - ...
%     1.828 * X(9) + 3.201 * X(5) + 5.854 * X(6) + 6.215 * X(7) - 3.36 *...
%     (1 - 1673./ T - log(T./1673)) - 7.01e-7 .* P./T + 1.54e-10 * (T - 1673).*P./T + ...
%     3.85e-17 * P.^2 ./ T;
% XFe2O3_FeO = exp(lnXFe2O3_FeO);

end

