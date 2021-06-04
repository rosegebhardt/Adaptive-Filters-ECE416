%% Rose Gebhardt: PSET4 Question 2
clear all; close all; clc; %#ok<CLALL>

%% Adaptive Equalization Setup

M = 21;
N = 500;

noise_var = 0.01;

s_options = [-1,1];
s = s_options(randi([1,2],N,1));

h_1 = [0.25,1,0.25];
h_2 = [0.25,1,-0.25];
h_3 = [-0.25,1,0.25];

mu_LMS = 0.05; mu_NLMS = 0.1;
num_runs = 100;
lam = 0.95; delta = 0.005;

%% Adaptive Equalization Results

[x_1,X_1] = form_X(h_1,s,noise_var,N,M);
[x_2,X_2] = form_X(h_2,s,noise_var,N,M);
[x_3,X_3] = form_X(h_3,s,noise_var,N,M);

[w_1_RLS,J_1_RLS] = learning_curve(lam,delta,s,X_1,num_runs);
[w_2_RLS,J_2_RLS] = learning_curve(lam,delta,s,X_2,num_runs);
[w_3_RLS,J_3_RLS] = learning_curve(lam,delta,s,X_3,num_runs);

w_1_wiener = wiener_filter(s,x_1,M);
e_1 = s(1+10:end-10) - w_1_wiener'*X_1;
Jmin_1 = mean(e_1.^2);

w_2_wiener = wiener_filter(s,x_2,M);
e_2 = s(1+10:end-10) - w_2_wiener'*X_2;
Jmin_2 = mean(e_2.^2);

w_3_wiener = wiener_filter(s,x_3,M);
e_3 = s(1+10:end-10) - w_3_wiener'*X_3;
Jmin_3 = mean(e_3.^2);

figure(1)
subplot(2,3,1)
plot(J_1_RLS,'LineWidth',1)
title('Learning Curve (Case 1)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 

subplot(2,3,4)
plot(w_1_RLS,'LineWidth',1); hold on; 
plot(w_1_wiener,'LineWidth',1); hold off;
legend_filter = legend('RLS','Wiener Filter');
set(legend_filter,'Interpreter','latex');
title('Comparison of Tap Weight Vectors (Case 1)','interpreter','latex')
xlabel('Index','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([1,21]);

subplot(2,3,2)
plot(J_2_RLS,'LineWidth',1)
title('Learning Curve (Case 2)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 

subplot(2,3,5)
plot(w_2_RLS,'LineWidth',1); hold on; 
plot(w_2_wiener,'LineWidth',1); hold off;
legend_filter = legend('RLS','Wiener Filter');
set(legend_filter,'Interpreter','latex');
title('Comparison of Tap Weight Vectors (Case 2)','interpreter','latex')
xlabel('Index','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([1,21]);

subplot(2,3,3)
plot(J_3_RLS,'LineWidth',1)
title('Learning Curve (Case 3)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 

subplot(2,3,6)
plot(w_3_RLS,'LineWidth',1); hold on; 
plot(w_3_wiener,'LineWidth',1); hold off;
legend_filter = legend('RLS','Wiener Filter');
set(legend_filter,'Interpreter','latex');
title('Comparison of Tap Weight Vectors (Case 3)','interpreter','latex')
xlabel('Index','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([1,21]);

%% Case 1 - Set Up

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

%% Case 1 - Adaptive MVDR and GSC

C_a = null(S');
g = eye(L);

num_runs = 100;
w_MVDR1 = zeros(M,N,L,num_runs);
w_GSC1 = zeros(M,N,L,num_runs);
e_MVDR1 = zeros(N,L,num_runs);
e_GSC1 = zeros(N,L,num_runs);

for index = 1:num_runs  
    [w_MVDR1(:,:,:,index),e_MVDR1(:,:,index)] = MVDR(lam,delta,S,A);
    [w_GSC1(:,:,:,index),e_GSC1(:,:,index)] = GSC(lam,delta,S,A,g);
end

figure(2)
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

figure(3)
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

figure(4)
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

figure(5)
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

lam = 0.95; delta = 0.005;

%% Case 2 - Adaptive MVDR and GSC

C_a = null(S');
g = eye(L);

num_runs = 100;
w_MVDR2 = zeros(M,N,L,num_runs);
w_GSC2 = zeros(M,N,L,num_runs);
e_MVDR2 = zeros(N,L,num_runs);
e_GSC2 = zeros(N,L,num_runs);

for index = 1:num_runs  
    [w_MVDR2(:,:,:,index),e_MVDR2(:,:,index)] = MVDR(lam,delta,S,A);
    [w_GSC2(:,:,:,index),e_GSC2(:,:,index)] = GSC(lam,delta,S,A,g);
end

figure(6)
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

figure(7)
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

figure(8)
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

figure(9)
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
