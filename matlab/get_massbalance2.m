function [PO2,FFeO1_5,mark,nFeO1_5] = get_massbalance2(Tm,Patm,Mmo,MO_mo,Xi,FeOt,g,Rp)
% Calculate mass balance for O
% I'm not sure if I will need Fe3_Fet yet....

muO = 15.9994e-3;                 % atomic weight of O in (kg/mole)
muFeO1_5 = 159.689e-3 / 2;        %molecular weight of FeO1.5 (kg/mole)
muFeO = 71.845e-3;                %molecular weight of FeO (kg/mole)


nFeOt = FeOt * Mmo/muFeO;           %moles of total Fe in magma ocean
nO_t = MO_mo / muO;                 %moles of O in magma ocean + atm system



%range of nFeO1_5/nFeOt values to iterate across. If FeO1_5 is outside this
%range, set all FeOt to FeO1_5 and allow O2 to build up in the atmosphere
a = 0.00;
b = 1.0;


count = 0;
fa = getf(a,Tm,Xi,Patm,nO_t,nFeOt,g,Rp);
fb = getf(b,Tm,Xi,Patm,nO_t,nFeOt,g,Rp);
if sign(fa)*sign(fb) > 0
%     if t > 5.5e5;
%         nO_t
%     end
    nFeO1_5 = nO_t * 2;             %otherwise put all O into FeO1.5
    nFeO = nFeOt - nFeO1_5;         %FeO is remainder of total Fe
    nO_atm = 0.0;                   %no O in atmosphere
    if nFeO < 0;
        nFeO = 0;
        nFeO1_5 = nFeOt;
        nO_atm = nO_t - 0.5 * nFeOt;
    end
end



while count <= 500
    p = a + (b-a) / 2;
    fp = getf(p,Tm,Xi,Patm,nO_t,nFeOt,g,Rp);
    
    if fp == 0 || ((b - a)/2 < 1e-17 && fp < 0)
        nFeO1_5 = p * nFeOt;
        mark = 0;
        break
    end
    
    count = count + 1;
    
    if sign(fa) * sign(fp) > 0
        a = p;
        fa = fp;
    else
        b = p;
        fb = fp;
    end
    
end

if count >= 500
%     count
%     nFeO1_5 = 0;
    nFeO1_5 = p * nFeOt;
    mark = 1;
end

nO_atm = nO_t - 0.5 * nFeO1_5;  %remainder of O in atmosphere
PO2 = (nO_atm * muO) * g / (4 * pi * Rp^2);
% if count >=150
%     hold on;plot(t/1e6/3.15569e7,PO2/1e5,'o')
% end

FFeO1_5 = nFeO1_5 * muFeO1_5 / Mmo;

if isnan(Tm)
    PO2 = 0;
    FFeO1_5 = 0;
end
[fo2] = get_fO2(Tm,Patm,[Xi(1:10) nFeOt-2*nO_t 2*nO_t]);


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    function myfun = getf(y,Tm,Xi,Patm,nO_t,nFeOt,g,Rp)
        muO = 15.9994e-3;                 % atomic weight of O in (kg/mole)
%         y = nFeO1_5 / nFeOt

        %calculate the fO2 in equilibrium with the FeO1.5/FeO ratio
        fO2fun = exp((log(y / (1 - y)) - 1.1492e4./Tm + 6.675 + 2.243 * Xi(3) + 1.828*Xi(9) - ...
            3.201*Xi(5) - 5.854 * Xi(6) - 6.215 * Xi(7) + 3.36 * (1 - 1673./Tm - ...
            log(Tm/1673)) + 7.01e-7 * Patm ./ Tm + 1.54e-10 * (Tm - 1673) * Patm / ...
            Tm - 3.85e-17 * Patm^2 / Tm) / 0.196);
        PO2fun = (nO_t/nFeOt - 0.5 * y) * nFeOt * muO * g/(4 * pi * Rp^2);
        
%         %O'Neill et al. (2006) silicate melt. Have to change the
%         %composition file to use this
%         fO2fun = 10^(4 * log10(y/(1-y)) - 28144./Tm + 13.95 + 3905 * Xi(4) - ...
%             13359 * Xi(5)./Tm - 14858 * Xi(6)./Tm - 9805 * Xi(7)./Tm + 10906 *...
%             Xi(3)./Tm + 110971 * Xi(8)./Tm - 11952 * (Xi(9))./Tm + (33122./Tm - ...
%             5.24)*((1 + 0.241 * Patm/1e9)^0.75 - 1) + (39156./Tm - 6.17)*((1 + ...
%             0.132 * Patm/1e9)^0.75 - 1));
        
        myfun = fO2fun - PO2fun;
        
    end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

end
