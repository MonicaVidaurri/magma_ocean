%% get_OLR_fn.m
% calculate OLR with gray assumption for GJ1132b

function [OLR] = get_OLR_fn(g,Teq,Tsurf,psurf,tog_ana,tog_cp0)
% function [OLR,p,T] = get_OLR_fn(g,Teq,Tsurf,psurf,tog_ana,tog_cp0)
sigma = 5.67e-8;    % Stefan-Boltzmann constant [W/m2/K4]
Rstar = 8.314462;   % ideal gas constant [J/K/kg]
muH2O = 18.015;     % water molar mass [g/mol]
Tskin = Teq/2^0.25; % skin temperature [K]
% kappa = 0.01;       % mass absorption coefficient [m2/kg]
% kappa = sqrt(0.04*g/3/1.01325e5); 
kappa = 0.00001;
cosa  = 0.5;        % we make hemispheric mean approx.

% to be generalized if/when we add CO2
R      = Rstar/(muH2O/1e3);    % specific gas constant [J/kg/K]
cp     = 2000;                 % specific heat capacity at constant pressure [J/kg/K]
tauinf = kappa*psurf/(g*cosa); % optical depth []

if tog_ana==1 
    %% simple analytic expression
    OLR = sigma*Tsurf.^4*tauinf.^(-4*R/cp)*gamma(1 + 4*R/cp);
elseif tog_ana==2
    %% simple integration
    nLev = 1e2;

    if tog_cp0
        p  = logspace(0,log10(psurf),nLev);   % pressure [Pa]
        p  = sort(p,'descend');
        T  = Tsurf*(p/psurf).^(R/cp);         % temperature [K]
    else
        % variable cp
        % dQ = dU - dW
        % 0  = cvdT + pdv 
        % 0  = cvdT - dp/rho + d(p/rho) 
        % dp/rho = cpdT
        % dlnp = (cp/R)dlnT
        % p = ps*exp[int(cp/R)dlnT]
        cp_ar = load('cp_H2O.dat');
        T  = logspace(1,log10(Tsurf),nLev);   % temperature [K]
        T  = sort(T,'descend');
        for iLev=1:nLev
            cp(iLev) = interp1(cp_ar(:,1),cp_ar(:,2),T(iLev),'linear','extrap')*1e3;
        end
        p = psurf*exp(cumtrapz(log(T),cp/R)); % pressure [Pa]
    end
    
    T(T<Tskin) = Tskin;                           % fix to isothermal stratosphere @ Tskin
    tau        = kappa*p/(g*cosa);                % optical depth []
    Trans      = exp(-tau);                       % transmissivity []
    Bsurf      = sigma*Tsurf^4;                   % surface blackbody radiation [W/m2]
    B          = sigma*T.^4;                      % blackbody radiation [W/m2]
    OLR        = Bsurf*Trans(1) + trapz(Trans,B); % outgoing longwave [W/m2]
    
end

return
