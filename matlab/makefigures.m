%make figures

%plot O2 abundance in bars (output is in Pa)
figure;loglog(t/1e6,PO2/1e5)
hold on;plot(t2/1e6,PO2_2/1e5,t3/1e6,PO2_3/1e5);
xlabel('time (Myr)')
ylabel('P_{O_2} (bars)')

%plot H2O abundance in atmosphere in bars
figure;loglog(t/1e6,Patm/1e5)
hold on;plot(t2/1e6,Patm_2/1e5,t3/1e6,Patm_3/1e5);
xlabel('time (Myr)')
ylabel('P_{H_2O} (bars)')
