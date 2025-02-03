



semilogy(wspeed,'linewidth',2,'color','r');

hold on

semilogy(gaspeed);

hold on

semilogy(hsspeed);

hold on


semilogy(bbspeed);

hold on

semilogy(despeed);

hold on

semilogy(ffspeed);

hold on

semilogy(iwospeed);

hold on

semilogy(psospeed);

hold on

semilogy(saspeed);

hold on


semilogy(sflaspeed);
hold on

% semilogy(tlbospeed);
% hold on



axis tight
grid on
box on


xlabel('Iteration');
ylabel('minimum');


legend('TSR','GA','HS','ICA','GWO','MFO','MVO','PSO','SCA','WOA')
% title('Speed reducer problem')
% ,'SPO'
