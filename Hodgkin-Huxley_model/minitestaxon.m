%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% testaxon.m 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% common time step 
dt = 0.005; % [us]

%% Fig 2: myelinated axon (intracellular current injection)
% morphological parameters
Ndef = 141; % number of nodes
Ddef = 2.0; % [um] axonal diameter 
Ldef = 2.0; % [um] nodal length 
Idef = 200.0; % [um] internodal length 

  % time vector 
Tleng = 20; % simulation time length [ms]
TstimS = 5; % current (Iinj) stimulus starting time [ms]
TstimE = 6; % current stimulus ending time [ms] 
t = 0:dt:Tleng; % time vector (real time value for each step)

% input vector (injecting a current of [iamp] Ampers
iamp = 100; % [pA]
Iinj = zeros(Ndef, length(t));
Iinj(20,(t>=TstimS)) = iamp; % [pA]
Iinj(20,(t>=TstimE)) = 0; % stop current stimulus

% calling the models
vWB = axonWBintra(Iinj,dt,Ndef,Ddef,Ldef,Idef);
% calculating spike conduction velocity: 
[m,i1] = max(vWB(40,:)); % node #40 and node #90
[m,i2] = max(vWB(90,:)); % times at which the membrane potential reached its peak
vel_mWB = ((Ldef+Idef)*50/1000)/(t(i2)-t(i1)); %  divided the distance between the two nodes by the travel time
s =  0:Tleng/dt; % number of steps
sp = t*vel_mWB; % [mm]

% plotting
##figure(2); 
##set(gcf,'Position', [200, 400, 560*1.5, 420]);
[m,i1] = max(vWB(20,:)); % find action potential in figure 2
position = t(i1)*vel_mWB


cla; hold on; 

figure(1)
plot(t-TstimS,vWB(20,:),'-','color',[0.5,0.5,0.5]); % #20 was orginnally the first node
##plot(t-TstimS,vWB(10,:)-10,'r-'); % #10 is simultaneous to #30 
plot(t-TstimS,vWB(30,:)-10,'r-'); 
plot(t-TstimS,vWB(40,:)-20,'y-'); 
plot(t-TstimS,vWB(50,:)-30,'g-'); 
plot(t-TstimS,vWB(60,:)-40,'c-'); 
plot(t-TstimS,vWB(70,:)-50,'b-'); 
plot(t-TstimS,vWB(80,:)-60,'m-'); 
plot(t-TstimS,vWB(90,:)-70,'k-'); 
xlim([-1,+6]);
ylim([-160,40]);
ylabel('potential [mV]');
xlabel('time (after first stimulus) [ms]')
title(sprintf('WB model (myelinated axon): %.2f [m/s]',vel_mWB));

figure(2)
##plot(sp,vWB(20,:),'-','color',[0.5,0.5,0.5]);
hold on;
plot(position,(-80:40)) % print vertical line to find a specific action potential
plot(sp,vWB(20,:)-10,'r-'); 
plot(sp,vWB(60,:)-20,'y-'); 
plot(sp,vWB(80,:)-30,'g-'); 
xlabel('distance [mm]');
ylabel('potential [mV]');
title(sprintf('impulse propagation along the axon: %.2f [m/s]',vel_mWB));