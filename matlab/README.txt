- Patm is atmospheric pressure of water vapor	
- PO2 is atmospheric O2	
- For new stars: use stellar tables by Baraffe et al 2015: http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/	
	a. Go to tracks+structure link
	b. Find the mass closest to the star you're trying to use, copy relevant columns which are time and L/Ls. 
	   These are actually log values, so will need to do 10^[value] for each column value. 
- This model doesn't handle condensation well, so for a habitable planet most water should condense out and	
  water vapor should be some fraction of saturated vapor pressure, dependent on surface pressure. Use an
  equation from here https://en.wikipedia.org/wiki/Vapour_pressure_of_water to find vapor pressure at any	
  surface temperature and use that as starting atmospheric water vapor abundance	
- Surface temp is the last column of the Tr variables (so like Tr(:,7), Tr2(:,5), Tr3(:,4))"	
- MO_mo is the mass of oxygen in the whole system (mantle + atm)	
- for printing out heating stuff in output:
    i. the block m_heatflux is used for each time stage in magma_ocean_evolution
    ii. consists of function from mantleheatflux: mantleheatflux(Tm,Ts,rs,Rp,Rc,g,rho_m,FH2O,meltfrac)
    iii. Tm = temp mantle; Ts = surf temp; rs = radius of solidification

