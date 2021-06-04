%% Rose Gebhardt: PSET3 Question 3
clear all; close all; clc; %#ok<CLALL>

%% Case 1 - Set Up

N = 250;
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

[~,D] = eig(R); lam_max = max(diag(D)); mu_max = 2/lam_max;
mu = 0.025*mu_max;
[M,L] = size(S);

%% Case 1 - Adaptive MVDR and GSC

C_a = null(S');
g = eye(L);

num_runs = 100;
w_MVDR1 = zeros(M,N,L,num_runs);
w_GSC1 = zeros(M,N,L,num_runs);
e_MVDR1 = zeros(N,L,num_runs);
e_GSC1 = zeros(N,L,num_runs);

for index = 1:num_runs  
    [w_MVDR1(:,:,:,index),e_MVDR1(:,:,index)] = MVDR(mu,S,A);
    [w_GSC1(:,:,:,index),e_GSC1(:,:,index)] = GSC(mu,S,A,g);
end

figure(1)
for index = 1:L
    subplot(L,2,2*index-1)
    e_m = mean(e_MVDR1,3);
    plot(abs(e_m(:,index)).^2,'LineWidth',1);
    title('MVDR Learning Curve','interpreter','latex')
    xlabel('Iterations','interpreter','latex'); 
    ylabel('E[n]','interpreter','latex');
    subplot(L,2,2*index)
    e_g = mean(e_GSC1,3);   
    plot(abs(e_g(:,index)).^2,'LineWidth',1);  
    title('GSC Learning Curve','interpreter','latex')
    xlabel('Iterations','interpreter','latex'); 
    ylabel('E[n]','interpreter','latex');
end

%% Case 1 - Mean-Sqared Deviation

w_mvdr1 = zeros(M,L);
w_q1 = zeros(M,L);
w_aopt1 = zeros(size(C_a,2),L);
w_gsc1 = zeros(M,L);

for index = 1:L
    w_mvdr1(:,index) = (pinv(R)*S(:,index))/(S(:,index)'*pinv(R)*S(:,index));
    w_q1(:,index) = S*pinv(S'*S)*g(:,index);
    w_aopt1(:,index) = pinv(C_a'*R*C_a)*C_a'*R*w_q1(:,index);
    w_gsc1(:,index) = w_q1(:,index)-C_a*w_aopt1(:,index);
end

avg_MVDR1 = mean(w_MVDR1,4);
avg_mvdr1 = repmat(w_mvdr1,1,1,N); avg_mvdr1 = permute(avg_mvdr1,[1 3 2]);

avg_GSC1 = mean(w_GSC1,4);
avg_gsc1 = repmat(w_gsc1,1,1,N); avg_gsc1 = permute(avg_gsc1,[1 3 2]);

delta_mvdr1 = avg_MVDR1 - avg_mvdr1;
D_mvdr1 = zeros(N,3);

delta_gsc1 = avg_GSC1 - avg_gsc1;
D_gsc1 = zeros(N,3);

for index = 1:L
    D_mvdr1(:,index) = vecnorm(delta_mvdr1(:,:,index)).';
    D_mvdr1(:,index) = D_mvdr1(:,index).^2;
    D_gsc1(:,index) = vecnorm(delta_gsc1(:,:,index)).';
    D_gsc1(:,index) = D_gsc1(:,index).^2;
end

figure(2)
subplot(1,2,1)
plot(D_mvdr1,'LineWidth',1)
legend_mvdr = legend('Source 1','Source 2','Source 3');
set(legend_mvdr,'Interpreter','latex');
title('MVDR Mean-Squared Deviation','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 
subplot(1,2,2)
plot(D_gsc1,'LineWidth',1)
legend_gsc = legend('Source 1','Source 2','Source 3');
set(legend_gsc,'Interpreter','latex');
title('GSC Mean-Squared Deviation','interpreter','latex')
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

array_MVDR1 = zeros(gridsize,gridsize,L); array_mvdr1 = zeros(gridsize,gridsize,L);
array_GSC1 = zeros(gridsize,gridsize,L); array_gsc1 = zeros(gridsize,gridsize,L); 

for index = 1:L
    array_MVDR1(:,:,index) = reshape(abs(w_MVDR1(:,end,index,1)'*S_range).^2,gridsize,gridsize);
    array_GSC1(:,:,index) = reshape(abs(w_GSC1(:,end,index,1)'*S_range).^2,gridsize,gridsize);
    array_mvdr1(:,:,index) = reshape(abs(w_mvdr1(:,index)'*S_range).^2,gridsize,gridsize);
    array_gsc1(:,:,index) = reshape(abs(w_gsc1(:,index)'*S_range).^2,gridsize,gridsize);
end

figure(3)
for index = 1:L
    subplot(3,2,2*index-1)
    surf(rad2deg(theta),rad2deg(phi),array_MVDR1(:,:,index),'EdgeColor','interp');
    title(['Adaptive MVDR Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('$\phi$','interpreter','latex');
    zlabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,180]);
    subplot(3,2,2*index)
    surf(rad2deg(theta),rad2deg(phi),array_GSC1(:,:,index),'EdgeColor','interp');
    title(['Adaptive GSC Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('$\phi$','interpreter','latex');
    zlabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,180]);
end

figure(4)
for index = 1:L
    subplot(3,2,2*index-1)
    surf(rad2deg(theta),rad2deg(phi),array_mvdr1(:,:,index),'EdgeColor','interp');
    title(['Ideal MVDR Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('$\phi$','interpreter','latex');
    zlabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,180]);
    subplot(3,2,2*index)
    surf(rad2deg(theta),rad2deg(phi),array_gsc1(:,:,index),'EdgeColor','interp');
    title(['Ideal GSC Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('$\phi$','interpreter','latex');
    zlabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,180]);
end
    
%% Case 1 - Attenuation

loc1 = [10,20,30]; loc2 = [20,160,150];

attenuation_MVDR1 = zeros(3,3);
for index = 1:3
    for jndex = 1:3
        attenuation_MVDR1(index,jndex) = 10*log10(array_MVDR1(loc1(index),loc2(index),jndex));
    end
end

attenuation_GSC1 = zeros(3,3);
for index = 1:3
    for jndex = 1:3
        attenuation_GSC1(index,jndex) = 10*log10(array_GSC1(loc1(index),loc2(index),jndex));
    end
end

attenuation_mvdr1 = zeros(3,3);
for index = 1:3
    for jndex = 1:3
        attenuation_mvdr1(index,jndex) = 10*log10(array_mvdr1(loc1(index),loc2(index),jndex));
    end
end

attenuation_gsc1 = zeros(3,3);
for index = 1:3
    for jndex = 1:3
        attenuation_gsc1(index,jndex) = 10*log10(array_gsc1(loc1(index),loc2(index),jndex));
    end
end

%% Case 2 - Set Up

N = 250;

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

%% Case 2 - Adaptive MVDR and GSC

C_a = null(S');
g = eye(L);

num_runs = 100;
w_MVDR2 = zeros(M,N,L,num_runs);
w_GSC2 = zeros(M,N,L,num_runs);
e_MVDR2 = zeros(N,L,num_runs);
e_GSC2 = zeros(N,L,num_runs);

for index = 1:num_runs  
    [w_MVDR2(:,:,:,index),e_MVDR2(:,:,index)] = MVDR(mu,S,A);
    [w_GSC2(:,:,:,index),e_GSC2(:,:,index)] = GSC(mu,S,A,g);
end

figure(5)
for index = 1:L
    subplot(L,2,2*index-1)
    e_m = mean(e_MVDR2,3);
    plot(abs(e_m(:,index)).^2,'LineWidth',1);
    title('MVDR Learning Curve','interpreter','latex')
    xlabel('Iterations','interpreter','latex'); 
    ylabel('E[n]','interpreter','latex');
    subplot(L,2,2*index)
    e_g = mean(e_GSC2,3);   
    plot(abs(e_g(:,index)).^2,'LineWidth',1);  
    title('GSC Learning Curve','interpreter','latex')
    xlabel('Iterations','interpreter','latex'); 
    ylabel('E[n]','interpreter','latex');
end

%% Case 2 - Mean-Sqared Deviation

w_mvdr2 = zeros(M,L);
w_q2 = zeros(M,L);
w_aopt2 = zeros(size(C_a,2),L);
w_gsc2 = zeros(M,L);

for index = 1:L
    w_mvdr2(:,index) = (pinv(R)*S(:,index))/(S(:,index)'*pinv(R)*S(:,index));
    w_q2(:,index) = S*pinv(S'*S)*g(:,index);
    w_aopt2(:,index) = pinv(C_a'*R*C_a)*C_a'*R*w_q2(:,index);
    w_gsc2(:,index) = w_q2(:,index)-C_a*w_aopt2(:,index);
end

avg_MVDR2 = mean(w_MVDR2,4);
avg_mvdr2 = repmat(w_mvdr2,1,1,N); avg_mvdr2 = permute(avg_mvdr2,[1 3 2]);

avg_GSC2 = mean(w_GSC2,4);
avg_gsc2 = repmat(w_gsc2,1,1,N); avg_gsc2 = permute(avg_gsc2,[1 3 2]);

delta_mvdr2 = avg_MVDR2 - avg_mvdr2;
D_mvdr2 = zeros(N,3);

delta_gsc2 = avg_GSC2 - avg_gsc2;
D_gsc2 = zeros(N,3);

for index = 1:L
    D_mvdr2(:,index) = vecnorm(delta_mvdr2(:,:,index)).';
    D_mvdr2(:,index) = D_mvdr2(:,index).^2;
    D_gsc2(:,index) = vecnorm(delta_gsc2(:,:,index)).';
    D_gsc2(:,index) = D_gsc2(:,index).^2;
end

figure(6)
subplot(1,2,1)
plot(D_mvdr2,'LineWidth',1)
legend_mvdr = legend('Source 1','Source 2','Source 3');
set(legend_mvdr,'Interpreter','latex');
title('MVDR Mean-Squared Deviation','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 
subplot(1,2,2)
plot(D_gsc2,'LineWidth',1)
legend_gsc = legend('Source 1','Source 2','Source 3');
set(legend_gsc,'Interpreter','latex');
title('GSC Mean-Squared Deviation','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex');

%% Case 2 - Array Pattern

arraysize = 1000;
theta_range = linspace(0,pi,arraysize);
S_range = steering_data2(theta_range,r,ones(1,N),d_lam,noise_var);

array_MVDR2 = zeros(arraysize,L); array_mvdr2 = zeros(arraysize,L);
array_GSC2 = zeros(arraysize,L); array_gsc2 = zeros(arraysize,L); 

for index = 1:L
    array_MVDR2(:,index) = abs(w_MVDR2(:,end,index,1)'*S_range).^2;
    array_GSC2(:,index) = abs(w_GSC2(:,end,index,1)'*S_range).^2;
    array_mvdr2(:,index) = abs(w_mvdr2(:,index)'*S_range).^2;
    array_gsc2(:,index) = abs(w_gsc2(:,index)'*S_range).^2;
end

figure(7)
for index = 1:L
    subplot(3,2,2*index-1)
    plot(rad2deg(theta_range),array_MVDR2(:,index),'LineWidth',1);
    title(['Adaptive MVDR Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,1.5]);
    subplot(3,2,2*index)
    plot(rad2deg(theta_range),array_GSC2(:,index),'LineWidth',1);
    title(['Adaptive GSC Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,1.5]);
end

figure(8)
for index = 1:L
    subplot(3,2,2*index-1)
    plot(rad2deg(theta_range),array_mvdr2(:,index),'LineWidth',1);
    title(['Ideal MVDR Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,1.5]);
    subplot(3,2,2*index)
    plot(rad2deg(theta_range),array_gsc2(:,index),'LineWidth',1);
    title(['Ideal GSC Array Pattern ' num2str(index)],'interpreter','latex')
    xlabel('$\theta$','interpreter','latex'); 
    ylabel('Magnitude','interpreter','latex');
    xlim([0,180]); ylim([0,1.5]);
end

%% Case 2 - Attenuation

[~,index_10] = min(abs(theta_range-Theta(1)));
[~,index_30] = min(abs(theta_range-Theta(2)));
[~,index_50] = min(abs(theta_range-Theta(3)));

loc = [index_10,index_30,index_50];

attenuation_MVDR2 = zeros(3,3);
for index = 1:3
    for jndex = 1:3
        attenuation_MVDR2(index,jndex) = 10*log10(array_MVDR2(loc(index),jndex));
    end
end

attenuation_GSC2 = zeros(3,3);
for index = 1:3
    for jndex = 1:3
        attenuation_GSC2(index,jndex) = 10*log10(array_GSC2(loc(index),jndex));
    end
end

attenuation_mvdr2 = zeros(3,3);
for index = 1:3
    for jndex = 1:3
        attenuation_mvdr2(index,jndex) = 10*log10(array_mvdr2(loc(index),jndex));
    end
end

attenuation_gsc2 = zeros(3,3);
for index = 1:3
    for jndex = 1:3
        attenuation_gsc2(index,jndex) = 10*log10(array_gsc2(loc(index),jndex));
    end
end

%% Comments

% All the beamformers for the second case turned out as expected. The
% magnitude of the result was near one at the source and attenuated well
% everywhere else. 

% The beamformer for source 1 and 3 turned out well for case 1, but case 1
% source 2 looked strange. There was a peak near the source, but not all
% the values away from the source were attenuated. This may have been
% because the negative value of theta in this case was handled incorrectly.

%% FF Comments

% nice coding; I liked how your MVDR/GSC could handle multiple cases at
% once (at first when you made g square it seemed weird, but I figured out
% what your were doing

% About some strange results: if the noise is low enough, the MVDR
% emphasizes attenuating the interferences and that sometimes means large
% lobes-- above 0dB-- form in places where there are no sources.
% The other thing you should have observed is that in general GSC has less
% attenuation for background regions than MVDR, because forcing nulls is
% like an extreme version of the comments I made for MVDR
