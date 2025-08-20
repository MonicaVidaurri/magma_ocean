
function [X] = get_comp(param)


%molecular weight
MW_SiO2 = 60.084;
MW_TiO2 = 79.866;
MW_Al2O3 = 101.961;
MW_FeO = 71.845;
MW_FeO1_5 = 159.689/2;
MW_MgO = 40.304;
MW_CaO = 56.078;
MW_Na2O = 61.979;
MW_K2O = 94.196;
MW_P2O5 = 141.945;


if param ==1
    %composition by mass
    %Bulk Silicate Earth
    SiO2 = 0.4597;
    TiO2 = 0.012;
    Al2O3 = 0.0477;
    FeOt = 0.08;
    MgO = 0.3666;
    CaO = 0.0378;
    Na2O = 0.0035;
    K2O = 0.0004;
    P2O5 = 0.002;
    Fe3_Fet = 1e-12;
    
end
    %moles of oxide
    nSiO2 = SiO2/MW_SiO2;
    nTiO2 = TiO2/MW_TiO2;
    nAl2O3 = Al2O3/MW_Al2O3;
%     nAlO1_5 = 2*nAl2O3;
    nFeOt = FeOt/MW_FeO;
    nMgO = MgO/MW_MgO;
    nCaO = CaO/MW_CaO;
    nNa2O = Na2O/MW_Na2O;
%     nNaO1_5 = 2*nNa2O;
    nK2O = K2O/MW_K2O;
%     nKO1_5 = 2*nK2O;
    nP2O5 = P2O5/MW_P2O5;
%     nPO2_5 = 2*nP2O5;

    total_moles = nSiO2 + nTiO2 + nAl2O3 + nFeOt + nMgO + nCaO + nNa2O + ...
        nK2O + nP2O5;

    %mole fractions of oxides
    X(1) = nSiO2 / total_moles;                 % XSiO2
    X(2) = nTiO2/ total_moles;                  % XTiO2
    X(3) = nAl2O3 / total_moles;                % XAl2O3 
    X(4) = nMgO/ total_moles;                   % XMgO 
    X(5) = nCaO/ total_moles;                   % XCaO 
    X(6) = nNa2O / total_moles;                 % XNa2O 
    X(7) = nK2O/ total_moles;                   % XK2O 
    X(8) = nP2O5/ total_moles;                  % XP2O5 
    X(9) = nFeOt / total_moles;                 % XFeOt
    X(10) = X(9) * Fe3_Fet;                     % XFeO1_5
    X(11) = X(9) - X(10);                       % XFeO
    X(12) = X(10);                              % XFe2O3 


end
