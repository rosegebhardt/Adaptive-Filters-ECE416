%% Rose Gebhardt: PSET4 Question 4
clear all; close all; clc; %#ok<CLALL>

%% Case 1 - Setup

N = 200;
d_lam = 0.5;
m = -10:10;

theta = deg2rad([10,20,30]); phi = deg2rad([20,-20,150]);
Theta = [theta;phi];

r0x = [1;0;0]; r0y = [0;1;0];
rx = m.*r0x; ry = m.*r0y; r0 = [rx,ry];
[r,~,~] = unique(r0.','rows');

beta = repmat([1;(10^(-5/10));(10^(-10/10))],[1,N]);
noise_var = 10^(-20/10);

[S,A] = steering_data1(Theta,r,beta,d_lam,noise_var);
R = S*diag(beta(:,1))*S' + noise_var*eye(length(S));

[M,L] = size(S);

lam = 0.95; delta = 0.005;

%% Case 1 - Adaptive MVDR 

num_runs = 100;
w_QRDRLS1 = zeros(M,N,L,num_runs); w_invQRDRLS1 = zeros(M,N,L,num_runs);
e_QRDRLS1 = zeros(N,L,num_runs); e_invQRDRLS1 = zeros(N,L,num_runs);

for index = 1:num_runs  
    [w_QRDRLS1(:,:,:,index),e_QRDRLS1(:,:,index)] = MVDR(lam,delta,S,A,'QRDRLS');
    [w_invQRDRLS1(:,:,:,index),e_invQRDRLS1(:,:,index)] = MVDR(lam,delta,S,A,'Inverse QRDRLS');
end

figure(1)
for index = 1:L
    subplot(2,L,index)
    e_q = mean(e_QRDRLS1,3);
    plot(abs(e_q(:,index)).^2,'LineWidth',1);
    title(['MVDR Learning Curve (QRD-RLS, Source ' num2str(index) ')'],'interpreter','latex')
    xlabel('Iterations','interpreter','latex'); 
    ylabel('E[n]','interpreter','latex');
    
    subplot(2,L,index+3)
    e_i = mean(e_invQRDRLS1,3);   
    plot(abs(e_i(:,index)).^2,'LineWidth',1);  
    title(['MVDR Learning Curve (Inverse QRD-RLS, Source ' num2str(index) ')'],'interpreter','latex')
    xlabel('Iterations','interpreter','latex'); 
    ylabel('E[n]','interpreter','latex');
end

%% Case 1 - Mean-Sqared Deviation

w_mvdr1 = zeros(M,L);

for index = 1:L
    w_mvdr1(:,index) = (pinv(R)*S(:,index))/(S(:,index)'*pinv(R)*S(:,index));
end

avg_QRDRLS1 = mean(w_QRDRLS1,4);
avg_invQRDRLS1 = mean(w_invQRDRLS1,4);
avg_mvdr1 = repmat(w_mvdr1,1,1,N); avg_mvdr1 = permute(avg_mvdr1,[1 3 2]);

delta_QRDRLS1 = avg_QRDRLS1 - avg_mvdr1;
delta_invQRDRLS1 = avg_invQRDRLS1 - avg_mvdr1;

D_QRDRLS1 = zeros(N,3);
D_invQRDRLS1 = zeros(N,3);

for index = 1:L
    D_QRDRLS1(:,index) = vecnorm(delta_QRDRLS1(:,:,index)).';
    D_QRDRLS1(:,index) = D_QRDRLS1(:,index).^2;
    D_invQRDRLS1(:,index) = vecnorm(delta_invQRDRLS1(:,:,index)).';
    D_invQRDRLS1(:,index) = D_invQRDRLS1(:,index).^2;
end

figure(2)
subplot(1,2,1)
plot(D_QRDRLS1,'LineWidth',1)
legend_qrdrls = legend('Source 1','Source 2','Source 3');
set(legend_qrdrls,'Interpreter','latex');
title('MVDR Mean-Squared Deviation (QRD-RLS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 
subplot(1,2,2)
plot(D_invQRDRLS1,'LineWidth',1)
legend_invqrdrls = legend('Source 1','Source 2','Source 3');
set(legend_invqrdrls,'Interpreter','latex');
title('MVDR Mean-Squared Deviation (Inverse QRD-RLS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex');

%% Case 1 - Array Pattern

gridsize = 181;
theta_domain = linspace(0,deg2rad(180),gridsize);
phi_domain = linspace(0,deg2rad(180),gridsize);
[theta,phi] = meshgrid(theta_domain,phi_domain);
Theta_Phi = [theta(:).';phi(:).'];

S_range = zeros(M,gridsize^2);

for index = 1:length(Theta_Phi)
    S_range(:,index) = steering_data1(Theta_Phi(:,index),r,ones(1,N),d_lam,noise_var);
end

array_QRDRLS1 = zeros(gridsize,gridsize,L);
array_invQRDRLS1 = zeros(gridsize,gridsize,L);
array_mvdr1 = zeros(gridsize,gridsize,L);

for index = 1:L
    array_QRDRLS1(:,:,index) = reshape(abs(w_QRDRLS1(:,end,index,1)'*S_range).^2,gridsize,gridsize);
    array_invQRDRLS1(:,:,index) = reshape(abs(w_invQRDRLS1(:,end,index,1)'*S_range).^2,gridsize,gridsize);
    array_mvdr1(:,:,index) = reshape(abs(w_mvdr1(:,index)'*S_range).^2,gridsize,gridsize);
end

figure(3)
for index = 1:L
    subplot(3,2,2*index-1)
    surf(rad2deg(theta),rad2deg(phi),array_QRDRLS1(:,:,index),'EdgeColor','interp');
    title(['QRD-RLS Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('$\phi$','interpreter','latex');
    zlabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,180]);
    subplot(3,2,2*index)
    surf(rad2deg(theta),rad2deg(phi),array_invQRDRLS1(:,:,index),'EdgeColor','interp');
    title(['Inverse QRD-RLS Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('$\phi$','interpreter','latex');
    zlabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,180]);
end

figure(4)
for index = 1:L
    subplot(3,1,index)
    surf(rad2deg(theta),rad2deg(phi),array_mvdr1(:,:,index),'EdgeColor','interp');
    title(['Ideal MVDR Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('$\phi$','interpreter','latex');
    zlabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,180]);
end

%% Case 2 - Setup

N = 200;

d_lam = 0.5;
M = 20;
Theta = deg2rad([10,30,50]);
r0z = [0;0;1]; m = 0:M-1; r = (m.*r0z).';

beta = repmat([1;(10^(-5/10));(10^(-15/10))],[1,N]);
noise_var = 10^(-25/10);

[S,A] = steering_data2(Theta,r,beta,d_lam,noise_var);
R = S*diag(beta(:,1))*S' + noise_var*eye(length(S));

[~,D] = eig(R); lam_max = max(diag(D)); mu_max = 2/lam_max;
mu = 0.025*mu_max;
[M,L] = size(S);

lam = 0.95; delta = 0.005;

%% Case 2 - Adaptive MVDR 

num_runs = 100;
w_QRDRLS2 = zeros(M,N,L,num_runs); w_invQRDRLS2 = zeros(M,N,L,num_runs);
e_QRDRLS2 = zeros(N,L,num_runs); e_invQRDRLS2 = zeros(N,L,num_runs);

for index = 1:num_runs  
    [w_QRDRLS2(:,:,:,index),e_QRDRLS2(:,:,index)] = MVDR(lam,delta,S,A,'QRDRLS');
    [w_invQRDRLS2(:,:,:,index),e_invQRDRLS2(:,:,index)] = MVDR(lam,delta,S,A,'Inverse QRDRLS');
end

figure(5)
for index = 1:L
    subplot(2,L,index)
    e_q = mean(e_QRDRLS2,3);
    plot(abs(e_q(:,index)).^2,'LineWidth',1);
    title(['MVDR Learning Curve (QRD-RLS, Source ' num2str(index) ')'],'interpreter','latex')
    xlabel('Iterations','interpreter','latex'); 
    ylabel('E[n]','interpreter','latex');
    
    subplot(2,L,index+3)
    e_i = mean(e_invQRDRLS2,3);   
    plot(abs(e_i(:,index)).^2,'LineWidth',1);  
    title(['MVDR Learning Curve (Inverse QRD-RLS, Source ' num2str(index) ')'],'interpreter','latex')
    xlabel('Iterations','interpreter','latex'); 
    ylabel('E[n]','interpreter','latex');
end

%% Case 2 - Mean-Sqared Deviation

w_mvdr2 = zeros(M,L);

for index = 1:L
    w_mvdr2(:,index) = (pinv(R)*S(:,index))/(S(:,index)'*pinv(R)*S(:,index));
end

avg_QRDRLS2 = mean(w_QRDRLS2,4);
avg_invQRDRLS2 = mean(w_invQRDRLS2,4);
avg_mvdr2 = repmat(w_mvdr2,1,1,N); avg_mvdr2 = permute(avg_mvdr2,[1 3 2]);

delta_QRDRLS2 = avg_QRDRLS2 - avg_mvdr2;
delta_invQRDRLS2 = avg_invQRDRLS2 - avg_mvdr2;

D_QRDRLS2 = zeros(N,3);
D_invQRDRLS2 = zeros(N,3);

for index = 1:L
    D_QRDRLS2(:,index) = vecnorm(delta_QRDRLS2(:,:,index)).';
    D_QRDRLS2(:,index) = D_QRDRLS2(:,index).^2;
    D_invQRDRLS2(:,index) = vecnorm(delta_invQRDRLS2(:,:,index)).';
    D_invQRDRLS2(:,index) = D_invQRDRLS2(:,index).^2;
end

figure(6)
subplot(1,2,1)
plot(D_QRDRLS2,'LineWidth',1)
legend_qrdrls = legend('Source 1','Source 2','Source 3');
set(legend_qrdrls,'Interpreter','latex');
title('MVDR Mean-Squared Deviation (QRD-RLS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 
subplot(1,2,2)
plot(D_invQRDRLS2,'LineWidth',1)
legend_invqrdrls = legend('Source 1','Source 2','Source 3');
set(legend_invqrdrls,'Interpreter','latex');
title('MVDR Mean-Squared Deviation (Inverse QRD-RLS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex');

%% Case 2 - Array Pattern

arraysize = 1000;
theta_range = linspace(0,pi,arraysize);
S_range = steering_data2(theta_range,r,ones(1,N),d_lam,noise_var);

array_QRDRLS2 = zeros(arraysize,L);
array_invQRDRLS2 = zeros(arraysize,L);
array_mvdr2 = zeros(arraysize,L);

for index = 1:L
    array_QRDRLS2(:,index) = abs(w_QRDRLS2(:,end,index,1)'*S_range).^2;
    array_invQRDRLS2(:,index) = abs(w_invQRDRLS2(:,end,index,1)'*S_range).^2;
    array_mvdr2(:,index) = abs(w_mvdr2(:,index)'*S_range).^2;
end

figure(7)
for index = 1:L
    subplot(3,2,2*index-1)
    plot(rad2deg(theta_range),array_QRDRLS2(:,index),'LineWidth',1);
    title(['QRD-RLS Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,1.5]);
    subplot(3,2,2*index)
    plot(rad2deg(theta_range),array_invQRDRLS2(:,index),'LineWidth',1);
    title(['Inverse QRD-RLS Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,1.5]);
end

figure(8)
for index = 1:L
    subplot(3,1,index)
    plot(rad2deg(theta_range),array_mvdr2(:,index),'LineWidth',1);
    title(['Ideal MVDR Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,1.5]);
end

%% FF Comments

% For all the beamforming problems, better to show the array responses etc in
% decibel scale (10*log10( )) to see features more clearly

% With hindsight, you should have run fewer iterations, or graph the
% learning curve and MS deviation curve with just the first few iterations
% showing: in many cases convergence was so fast you really didn't see much
% on the curves

% Otherwise overall very good coding, good reuse of code and organization
% etc!!