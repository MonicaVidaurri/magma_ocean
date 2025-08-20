function [PO2,FFeO1_5,count,nFeO1_5] = get_massbalance4(Tm,Patm,Mmo,MO_mo,Xi,FeOt,gp,Rp)
% Calculate mass balance for O
% I'm not sure if I will need Fe3_Fet yet....

muO = 15.9994e-3;                 % atomic weight of O in (kg/mole)
muFeO1_5 = 159.689e-3 / 2;        %molecular weight of FeO1.5 (kg/mole)
muFeO = 71.845e-3;                %molecular weight of FeO (kg/mole)


nFeOt = FeOt * Mmo/muFeO;           %moles of total Fe in magma ocean
nO_t = MO_mo / muO;                 %moles of O in magma ocean + atm system

    count = 0;
if nO_t < 1e-3*nFeOt

    y0 = nFeOt*1e-3;

    while count <= 100

        y = getf(y0,Tm,Xi,Patm,nO_t,nFeOt,gp,Rp);

        if abs(y - y0) < 1e-12 && y > 0
            nFeO1_5 = y;
            nO_atm = nO_t - 0.5*nFeO1_5;
            PO2 = (nO_atm * muO) * gp / (4 * pi * Rp^2);
            FFeO1_5 = nFeO1_5 * muFeO1_5 / Mmo;
            break
        end

        count = count + 1;
        
        if count >= 50
            nFeO1_5 = nO_t*2;
            nO_atm = nO_t - 0.5*nFeO1_5;
            PO2 = (nO_atm * muO) * gp / (4 * pi * Rp^2);
            FFeO1_5 = nFeO1_5 * muFeO1_5 / Mmo;
            break
        end
        
        y0 = y;
        
    end

    if nFeO1_5 > nFeOt
        nFeO1_5
    end


else
    [PO2,FFeO1_5,mark,nFeO1_5] = get_massbalance2(Tm,Patm,Mmo,MO_mo,Xi,FeOt,gp,Rp);
    nO_atm = nO_t - 0.5*nFeO1_5;
        if nFeO1_5 > nFeOt
        nFeO1_5
    end

end

if PO2 == 0
    PO2 = get_fO2(Tm,Patm,[Xi(1:10) nFeOt-2*nO_t 2*nO_t]);
end





% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    function myfun = getf(y,Tm,Xi,Patm,nO_t,nFeOt,gp,Rp)
        muO = 15.9994e-3;                 % atomic weight of O in (kg/mole)
%         y = nFeO1_5

        %calculate the fO2 in equilibrium with the FeO1.5/FeO ratio
        myfun = 2 *(nO_t - 4*pi*Rp^2/muO/gp *exp(1/0.196 *(log(y/(nFeOt -y))...
            - 1.1492e4./Tm + 6.675 + 2.243 * Xi(3) + 1.828*Xi(9) - ...
            3.201*Xi(5) - 5.854 * Xi(6) - 6.215 * Xi(7) + 3.36 * (1 - 1673./Tm - ...
            log(Tm/1673)) + 7.01e-7 * Patm ./ Tm + 1.54e-10 * (Tm - 1673) * Patm / ...
            Tm - 3.85e-17 * Patm^2 / Tm)));
        
    end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
end
