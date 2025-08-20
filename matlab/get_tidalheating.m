function tidal_heating = get_tidalheating(t, mantle_temp, mantle_pressure, mantle_volume)
% %can skip mantle pressure if worse comes to worse
% Calculates tidal heating based on temperature, pressure, and volume of mantle material over time
% Constants
R = 8.31446262;
mantle_solidus = 1600.0;
mantle_liquidus = 2000.0;
liquid_shear = 1.0e-5;
premelt_shear = 55.0e9;
liq_viscosity_const = 1000.0;
crit_melt_frac = 0.5;
crit_melt_frac_width = 0.05;
hn_visc_slope_1 = 13.5;
hn_visc_falloff_slope = 370.0;
hn_shear_param_1 = 40000.0;
hn_shear_param_2 = 25.0;
hn_shear_falloff_slope = 700.0;
liq_reference_viscosity = 0.2;
liq_reference_temperature = 2000.0;
liq_molar_activation_energy = 6.64e-20;
liq_molar_activation_volume = 0.0;
sol_reference_viscosity = 1.0e22;
sol_reference_temperature = 1000.0;
sol_molar_activation_energy = 300000.0;
sol_molar_activation_volume = 0.0;

% Orbital and planetary properties
host_mass = 2.0e29;
target_radius = 0.920 * 6.371e6;
target_mass = 0.692 * 5.9722e24;
G = 6.6743e-11;
target_gravity = G * target_mass / target_radius^2;
target_volume = (4/3) * pi * target_radius^3;
target_bulk_density = target_mass / target_volume;
normal_eccentricity = 0.00510;
orbital_period = 6.101013 * 86400.0;
semi_major_axis = 0.02925 * 149597870700.0;
orbital_freq = 2.0 * pi / orbital_period;
planet_volume = (4/3) * pi * target_radius^3;
volume_frac = mantle_volume ./ planet_volume;

% Eccentricity oscillation
time_start = 1000;
time_end = 3000;
eccen_oscil_freq = 2.0 * pi / 1000.0;
eccen_phase = time_start;
eccen_offset = 0.05;
eccen_amplitude = 0.15;

eccentricity = max(zeros(size(time)), eccen_offset + eccen_amplitude .* sin(eccen_oscil_freq .* (time - eccen_phase)));
eccentricity(time < time_start) = 0.0;
eccentricity(time > time_end) = normal_eccentricity;

% Solid viscosity
sol_temp_diff = (1 ./ mantle_temp) - (1 / sol_reference_temperature);
sol_exponent = ((sol_molar_activation_energy + mantle_pressure .* sol_molar_activation_volume) ./ R) .* sol_temp_diff;
sol_viscosity = sol_reference_viscosity .* exp(sol_exponent);

% Liquid viscosity
liq_temp_diff = (1 ./ mantle_temp) - (1 / liq_reference_temperature);
liq_exponent = ((liq_molar_activation_energy + mantle_pressure .* liq_molar_activation_volume) ./ R) .* liq_temp_diff;
liq_viscosity_arr = liq_reference_viscosity .* exp(liq_exponent);

% Melt fraction
melt_frac = (mantle_temp - mantle_solidus) ./ (mantle_liquidus - mantle_solidus);
melt_frac = min(max(melt_frac, 0), 1);

crit_melt_frac_plus_width = crit_melt_frac + crit_melt_frac_width;
break_down_temp = mantle_solidus + crit_melt_frac * (mantle_liquidus - mantle_solidus);
premelt_viscosity = sol_viscosity;
premelt_shear_modulus = premelt_shear * ones(size(mantle_temp));

% Viscosity
viscosity = zeros(size(mantle_temp));
idx = melt_frac <= 0;
viscosity(idx) = premelt_viscosity(idx);
idx = (melt_frac > 0) & (melt_frac < crit_melt_frac);
viscosity(idx) = premelt_viscosity(idx) .* exp(-hn_visc_slope_1 .* melt_frac(idx));
idx = (melt_frac >= crit_melt_frac) & (melt_frac <= crit_melt_frac_plus_width);
viscosity(idx) = premelt_viscosity(idx) .* exp(-hn_visc_slope_1 * crit_melt_frac) .* exp(-hn_visc_falloff_slope .* (melt_frac(idx) - crit_melt_frac));
idx = melt_frac > crit_melt_frac_plus_width;
viscosity(idx) = liq_viscosity_const;
viscosity(viscosity < liq_viscosity_const) = liq_viscosity_const;

% Shear modulus
shear_modulus = zeros(size(mantle_temp));
idx = melt_frac <= 0;
shear_modulus(idx) = premelt_shear_modulus(idx);
idx = (melt_frac > 0) & (melt_frac < crit_melt_frac);
shear_modulus(idx) = premelt_shear_modulus(idx) .* exp((hn_shear_param_1 ./ mantle_temp(idx)) - hn_shear_param_2);
idx = (melt_frac >= crit_melt_frac) & (melt_frac <= crit_melt_frac_plus_width);
shear_modulus(idx) = premelt_shear_modulus(idx) .* ...
    exp((hn_shear_param_1 / break_down_temp) - hn_shear_param_2) .* ...
    exp(-hn_shear_falloff_slope .* (melt_frac(idx) - crit_melt_frac));
idx = melt_frac > crit_melt_frac_plus_width;
shear_modulus(idx) = liquid_shear;
shear_modulus(shear_modulus < liquid_shear) = liquid_shear;

% Rheology and complex Love number
compliance = 1.0 ./ shear_modulus;
complex_compliance = compliance .* (-1i) ./ (viscosity .* orbital_freq);
eff_rigid = (19 / 2) * premelt_shear ./ (target_gravity * target_radius * target_bulk_density);
rheology_factor = complex_compliance .* premelt_shear;
cmplx_love = (3 / 2) ./ (1 + (eff_rigid ./ rheology_factor));

% Tidal heating
tidal_susceptibility = (3 / 2) * G * host_mass^2 * target_radius^5 / semi_major_axis^6;
tidal_heating = volume_frac .* -imag(cmplx_love) * 7.0 .* tidal_susceptibility .* orbital_freq .* eccentricity.^2;

end