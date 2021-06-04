%% Rose Gebhardt - PSET4 Question 3
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

%% Adaptive Equalizer Results

[x_1,X_1] = form_X(h_1,s,noise_var,N,M);
[x_2,X_2] = form_X(h_2,s,noise_var,N,M);
[x_3,X_3] = form_X(h_3,s,noise_var,N,M);

[w_QRDRLS1,e_QRDRLS1] = QRDRLS(lam,delta,X_1,s,480);
[w_invQRDRLS1,e_invQRDRLS1] = inverseQRDRLS(lam,delta,X_1,s,480);

[w_QRDRLS2,e_QRDRLS2] = QRDRLS(lam,delta,X_2,s,480);
[w_invQRDRLS2,e_invQRDRLS2] = inverseQRDRLS(lam,delta,X_2,s,480);

[w_QRDRLS3,e_QRDRLS3] = QRDRLS(lam,delta,X_3,s,480);
[w_invQRDRLS3,e_invQRDRLS3] = inverseQRDRLS(lam,delta,X_3,s,480);

figure(1)

subplot(1,3,1)
plot(w_QRDRLS1,'LineWidth',1); hold on; 
plot(w_invQRDRLS1,'LineWidth',1); hold off;
legend_filter = legend('QRD-RLS','Inverse QRD-RLS');
set(legend_filter,'Interpreter','latex');
title('Comparison of Tap Weight Vectors (Case 1)','interpreter','latex')
xlabel('Index','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([1,21]);

subplot(1,3,2)
plot(w_QRDRLS2,'LineWidth',1); hold on; 
plot(w_invQRDRLS2,'LineWidth',1); hold off;
legend_filter = legend('QRD-RLS','Inverse QRD-RLS');
set(legend_filter,'Interpreter','latex');
title('Comparison of Tap Weight Vectors (Case 2)','interpreter','latex')
xlabel('Index','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([1,21]);

subplot(1,3,3)
plot(w_QRDRLS3,'LineWidth',1); hold on; 
plot(w_invQRDRLS3,'LineWidth',1); hold off;
legend_filter = legend('QRD-RLS','Inverse QRD-RLS');
set(legend_filter,'Interpreter','latex');
title('Comparison of Tap Weight Vectors (Case 3)','interpreter','latex')
xlabel('Index','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([1,21]);

%% Test BER

num_extend = 10000;
s_extend = s_options(randi([1,2],num_extend,1));

[x_1_extend,X_1_extend] = form_X(h_1,s_extend,noise_var,num_extend,M);
[x_2_extend,X_2_extend] = form_X(h_2,s_extend,noise_var,num_extend,M);
[x_3_extend,X_3_extend] = form_X(h_3,s_extend,noise_var,num_extend,M);

s_true = zeros(num_extend-M+1,1);

s_QRDRLS1 = zeros(num_extend-M+1,1);
s_QRDRLS2 = zeros(num_extend-M+1,1);
s_QRDRLS3 = zeros(num_extend-M+1,1);

s_invQRDRLS1 = zeros(num_extend-M+1,1);
s_invQRDRLS2 = zeros(num_extend-M+1,1);
s_invQRDRLS3 = zeros(num_extend-M+1,1);

for index = 1:num_extend-M+1
    
    s_true(index) = s_extend(index+10);
    
    s_QRDRLS1(index) = w_QRDRLS1'*X_1_extend(:,index);
    s_QRDRLS2(index) = w_QRDRLS2'*X_2_extend(:,index);
    s_QRDRLS3(index) = w_QRDRLS3'*X_3_extend(:,index);
    
    s_invQRDRLS1(index) = w_invQRDRLS1'*X_1_extend(:,index);
    s_invQRDRLS2(index) = w_invQRDRLS2'*X_2_extend(:,index);
    s_invQRDRLS3(index) = w_invQRDRLS3'*X_3_extend(:,index);
    
end

error_QRDRLS1 = mean(s_true ~= 1-2*(s_QRDRLS1<0));
error_QRDRLS2 = mean(s_true ~= 1-2*(s_QRDRLS2<0));
error_QRDRLS3 = mean(s_true ~= 1-2*(s_QRDRLS3<0));

error_invQRDRLS1 = mean(s_true ~= 1-2*(s_invQRDRLS1<0));
error_invQRDRLS2 = mean(s_true ~= 1-2*(s_invQRDRLS2<0));
error_invQRDRLS3 = mean(s_true ~= 1-2*(s_invQRDRLS3<0));

%% See Error with Random Data

% s_fake = s_options(randi([1,2],num_extend-M+1,1));
% 
% error_QRDRLS1 = mean(s_fake.' ~= 1-2*(s_QRDRLS1<0));
% error_QRDRLS2 = mean(s_fake.' ~= 1-2*(s_QRDRLS2<0));
% error_QRDRLS3 = mean(s_fake.' ~= 1-2*(s_QRDRLS3<0));
% 
% error_invQRDRLS1 = mean(s_fake.' ~= 1-2*(s_invQRDRLS1<0));
% error_invQRDRLS2 = mean(s_fake.' ~= 1-2*(s_invQRDRLS2<0));
% error_invQRDRLS3 = mean(s_fake.' ~= 1-2*(s_invQRDRLS3<0));
