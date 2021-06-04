%% Rose Gebhardt: PSET3 Question 2
clear all; close all; clc; %#ok<CLALL>

%% Setup

% FF comments: Nice job; showing tap weights probably better with a stem plot

M = 21;
N = 1000;

noise_var = 0.01;

s_options = [-1,1];
s = s_options(randi([1, 2],N,1));

h_1 = [0.25,1,0.25];
h_2 = [0.25,1,-0.25];
h_3 = [-0.25,1,0.25];

mu_LMS = 0.05; mu_NLMS = 0.1;
num_runs = 100;

%% Case 1

[x_1,X_1] = form_X(h_1,s,noise_var,N,M);

[w_1_LMS,J_1_LMS] = learning_curve(mu_LMS,s,X_1,'LMS',num_runs);
[w_1_NLMS,J_1_NLMS] = learning_curve(mu_NLMS,s,X_1,'NLMS',num_runs);

figure(1)
subplot(2,1,1)
plot(J_1_LMS,'LineWidth',1)
title('Learning Curve (Case 1, LMS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(2,1,2)
plot(J_1_NLMS,'LineWidth',1)
title('Learning Curve (Case 1, NLMS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex');

w_1_wiener = wiener_filter(s,x_1,M);
e_1 = s(1+10:end-10) - w_1_wiener'*X_1;
Jmin_1 = mean(e_1.^2);

figure(2)
plot(w_1_LMS,'LineWidth',1); hold on; 
plot(w_1_NLMS,'LineWidth',1); hold on; 
plot(w_1_wiener,'LineWidth',1); hold off;
legend_filter = legend('LMS','NLMS','Wiener Filter');
set(legend_filter,'Interpreter','latex');
title('Comparison of Computed Tap Weight Vectors','interpreter','latex')
xlabel('Index','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');

%% Case 2

[x_2,X_2] = form_X(h_2,s,noise_var,N,M);

[w_2_LMS,J_2_LMS] = learning_curve(mu_LMS,s,X_2,'LMS',num_runs);
[w_2_NLMS,J_2_NLMS] = learning_curve(mu_NLMS,s,X_2,'NLMS',num_runs);

figure(3)
subplot(2,1,1)
plot(J_2_LMS,'LineWidth',1)
title('Learning Curve (Case 2, LMS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(2,1,2)
plot(J_2_NLMS,'LineWidth',1)
title('Learning Curve (Case 2, NLMS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex');

w_2_wiener = wiener_filter(s,x_2,M);
e_2 = s(1+10:end-10) - w_2_wiener'*X_2;
Jmin_2 = mean(e_2.^2);

figure(4)
plot(w_2_LMS,'LineWidth',1); hold on; 
plot(w_2_NLMS,'LineWidth',1); hold on; 
plot(w_2_wiener,'LineWidth',1); hold off;
legend_filter = legend('LMS','NLMS','Wiener Filter');
set(legend_filter,'Interpreter','latex');
title('Comparison of Computed Tap Weight Vectors','interpreter','latex')
xlabel('Index','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');

%% Case 3

[x_3,X_3] = form_X(h_3,s,noise_var,N,M);

[w_3_LMS,J_3_LMS] = learning_curve(mu_LMS,s,X_3,'LMS',num_runs);
[w_3_NLMS,J_3_NLMS] = learning_curve(mu_NLMS,s,X_3,'NLMS',num_runs);

figure(5)
subplot(2,1,1)
plot(J_3_LMS,'LineWidth',1)
title('Learning Curve (Case 3, LMS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(2,1,2)
plot(J_3_NLMS,'LineWidth',1)
title('Learning Curve (Case 3, NLMS)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex');

w_3_wiener = wiener_filter(s,x_3,M);
e_3 = s(1+10:end-10) - w_3_wiener'*X_3;
Jmin_3 = mean(e_3.^2);

figure(6)
plot(w_3_LMS,'LineWidth',1); hold on; 
plot(w_3_NLMS,'LineWidth',1); hold on; 
plot(w_3_wiener,'LineWidth',1); hold off;
legend_filter = legend('LMS','NLMS','Wiener Filter');
set(legend_filter,'Interpreter','latex');
title('Comparison of Computed Tap Weight Vectors','interpreter','latex')
xlabel('Index','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
