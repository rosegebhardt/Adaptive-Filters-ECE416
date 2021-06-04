%% Rose Gebhardt: Kalman Project
clear all; close all; clc; %#ok<CLALL>

%% Set Up

rng(420);

A1 = [-0.9,1,0,0;0,-0.9,0,0;0,0,-0.5,0.5;1,0,-0.5,-0.5];
A2 = [0.9,1,0,0;0,0.9,0,0;0,0,-0.5,0.5;1,0,-0.5,-0.5];

C = [1,-1,1,-1;0,1,0,1];

Qv = [1/4,1/4,0,0;1/4,1/2,1/4,0;0,1/4,1/2,1/4;0,0,1/4,1/2];
Qw = 0.1*eye(2);

%% Studying the Prescribed System

eig_A1 = eig(A1); eig_A2 = eig(A2);
stable_A1 = prod(abs(eig_A1) < 1); stable_A2 = prod(abs(eig_A2) < 1);
geomult_A1 = rank(null(A1 + 0.9*eye(4))); geomult_A2 = rank(null(A2 - 0.9*eye(4)));

K_A1 = dlyap(A1, Qv); K_A2 = dlyap(A2, Qv);

p1 = max(abs(eig_A1)); p2 = max(abs(eig_A2));

N1 = 0; N2 = 0;
while p1^N1 > 0.01
    N1 = N1+1;
end
while p2^N2 > 0.01
    N2 = N2+1;
end

obsv_A1 = (4 == rank(obsv(A1,C))); obsv_A2 = (4 == rank(obsv(A2,C)));

[K1_ss,~,~] = dare(A1',C',Qv,Qw); [K2_ss,~,~] = dare(A2',C',Qv,Qw);

check_K1ss = prod(eig(K1_ss) > 0)*(issymmetric(K1_ss));
check_K2ss = prod(eig(K2_ss) > 0)*(issymmetric(K2_ss));

spec_norm = norm(K1_ss - K2_ss);

N = 50;
x_test = zeros(4,2*N);
x_test(:,1) = ones(4,1);
v = Qv*randn(4,2*N);

for jj = 2:N
    x_test(:,jj) = A1*x_test(:,jj-1) + v(:,jj);
end
for jj = N+1:2*N
    x_test(:,jj) = A2*x_test(:,jj-1) + v(:,jj);
end

figure(1)
plot(1:2*N,x_test,'Linewidth',1);
legend_test = legend('$x_1$','$x_2$','$x_3$','$x_4$','location','southwest');
set(legend_test,'Interpreter','latex');
title('Process as System Equations Switch','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

%% Covariance Kalman Filter Setup

num_runs = 100;
count1 = zeros(num_runs,1); count2 = zeros(num_runs,1);

for jj = 1:num_runs
    x1 = randn(4,1); x1_sys = x1;
    K1 = eye(4);
    while (norm(K1-K1_ss)/norm(K1_ss) > 0.01)
        y1 = C*x1_sys + Qw*rand(2,1);
        x1_sys = A1*x1_sys + Qv*randn(4,1);
        [~,x1,~,K1,~] = covarianceKalman(x1,K1,y1,A1,C,Qv,Qw);
        count1(jj) = count1(jj)+1;
    end
    x2 = randn(4,1); x2_sys = x2;
    K2 = eye(4);
    while (norm(K2-K2_ss)/norm(K2_ss) > 0.01)
        y2 = C*x2_sys + Qw*rand(2,1);
        x2_sys = A2*x2_sys + Qv*randn(4,1);
        [~,x2,~,K2,~] = covarianceKalman(x2,K2,y2,A2,C,Qv,Qw);
        count2(jj) = count2(jj)+1;
    end
end

if (mean(count1)<10)
    N1 = 10;
else
    N1 = ceil(mean(count1));
end

if (mean(count2)<10)
    N2 = 10;
else
    N2 = ceil(mean(count2));
end

%% Covariance Kalman Filter Results

num_trials = 5;
x_sys = zeros(4,N1+N2,num_trials);
x_pred = zeros(4,N1+N2,num_trials);
K_pred = zeros(4,4,N1+N2,num_trials);
x_est = zeros(4,N1+N2,num_trials);
K_est = zeros(4,4,N1+N2,num_trials);

for jj = 1:num_trials
    x_sys(:,1,jj) = randn(4,1); 
    x_pred(:,1,jj) = x_sys(:,1,jj); 
    K_pred(:,:,1,jj) = eye(4);
    for ii = 1:(N1-1)
        x_sys(:,ii+1,jj) = A1*x_sys(:,ii,jj) + Qv*randn(4,1);
        y = C*x_sys(:,ii,jj) + Qw*randn(2,1);
        [x_est(:,ii,jj),x_pred(:,ii+1,jj),K_est(:,:,ii,jj),K_pred(:,:,ii+1,jj),G] = ...
            covarianceKalman(x_pred(:,ii,jj),K_pred(:,:,ii,jj),y,A1,C,Qv,Qw);
    end
    for ii = N1:(N1+N2-1)
        x_sys(:,ii+1,jj) = A2*x_sys(:,ii,jj) + Qv*randn(4,1);
        y = C*x_sys(:,ii,jj) + Qw*randn(2,1);
        [x_est(:,ii,jj),x_pred(:,ii+1,jj),K_est(:,:,ii,jj),K_pred(:,:,ii+1,jj),G] = ...
            covarianceKalman(x_pred(:,ii,jj),K_pred(:,:,ii,jj),y,A2,C,Qv,Qw);
    end
end

figure(2)
plot(1:N1+N2,reshape(vecnorm(x_pred),[N1+N2,num_trials]),'Linewidth',1);
legend_cov = legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5','location','best');
set(legend_cov,'Interpreter','latex');
title('Magnitude of State (Covariance Kalman Filter)','interpreter','latex')
xlabel('Time ($n$)','interpreter','latex'); 
ylabel('Magnitude ($|x[n]|$)','interpreter','latex'); 

figure(3)
plot3(reshape(x_pred(1,1:N1,1),[N1,1]),...
      reshape(x_pred(2,1:N1,1),[N1,1]),...
      reshape(x_pred(3,1:N1,1),[N1,1]),'blue','Linewidth',1);
hold on;
plot3(reshape(x_pred(1,N1:N1+N2,1),[N2+1,1]),...
      reshape(x_pred(2,N1:N1+N2,1),[N2+1,1]),...
      reshape(x_pred(3,N1:N1+N2,1),[N2+1,1]),'red','Linewidth',1);
legend_cov3d = legend('A1','A2','location','best');
set(legend_cov3d,'Interpreter','latex');
title('Three Components of State (Covariance Kalman Filter)','interpreter','latex')
xlabel('$x_1[n]$','interpreter','latex'); 
ylabel('$x_2[n]$','interpreter','latex'); 
zlabel('$x_3[n]$','interpreter','latex');

%% Covariance Kalman Filter Analysis

norm_deltax_pred = reshape(vecnorm(x_sys-x_pred),[N1+N2,num_trials]);
norm_deltax_est = reshape(vecnorm(x_sys-x_est),[N1+N2,num_trials]);

norm_deltaK_pred = zeros(N1+N2,1);
for jj = 1:N1
    norm_deltaK_pred(jj) = norm(K_pred(:,:,jj,1)-K1_ss);
end
for jj = N1+1:N2
    norm_deltaK_pred(jj) = norm(K_pred(:,:,jj,1)-K2_ss);
end

norm_K_pred = zeros(N1+N2,1);
for jj = 1:N1+N2
    norm_K_pred(jj) = norm(K_pred(:,:,jj,1));
end

norm_K_est = zeros(N1+N2,1);
for jj = 1:N1+N2
    norm_K_est(jj) = norm(K_est(:,:,jj,1));
end

norm_deltaK_est = zeros(N1+N2-1,1);
for jj = 1:N1+N2-1
    norm_deltaK_est(jj) = norm(K_est(:,:,jj+1,1)-K_est(:,:,jj,1));
end

f4 = figure(4);
f4.Name = "Covariance Kalman Filter";

subplot(2,3,1)
plot(1:N1+N2,norm_deltax_pred,'Linewidth',1);
legend_cov = legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5','location','best');
set(legend_cov,'Interpreter','latex');
title('$\| x(n) - \hat{x}(n|n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,2)
plot(1:N1+N2,norm_deltax_est,'Linewidth',1);
legend_cov = legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5','location','best');
set(legend_cov,'Interpreter','latex');
title('$\| x(n) - \hat{x}(n|n) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([1,N1+N2-1])

subplot(2,3,3)
plot(1:N1+N2,norm_deltaK_pred,'Linewidth',1);
title('$\| K(n|n-1) - K_{i,ss} \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,4)
plot(1:N1+N2,norm_K_pred,'Linewidth',1);
title('$\| K(n|n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,5)
plot(1:N1+N2,norm_K_est,'Linewidth',1);
title('$\| K(n,n) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,6)
plot(1:N1+N2-1,norm_deltaK_est,'Linewidth',1);
title('$\| K(n,n)-K(n-1,n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

%% Information Kalman Filter Setup

num_runs = 100;
count1 = zeros(num_runs,1); count2 = zeros(num_runs,1);

for ii = 1:num_runs
    
    K1 = eye(4); P1 = pinv(K1);
    x1 = randn(4,1); chi1 = P1*x1;
    x1_sys = x1;
    
    while (norm(K1-K1_ss)/norm(K1_ss) > 0.01)
        y1 = C*x1_sys + Qw*rand(2,1);
        x1_sys = A1*x1_sys + Qv*randn(4,1);
        [~,chi1,~,P1] = informationKalman(chi1,P1,y1,A1,C,Qv,Qw);
        count1(ii) = count1(ii)+1;
        K1 = pinv(P1);
    end
    
    K2 = eye(4); P2 = pinv(K2);
    x2 = randn(4,1); chi2 = P2*x2;
    x2_sys = x2;
    
    while (norm(K2-K2_ss)/norm(K2_ss) > 0.01)
        y2 = C*x2_sys + Qw*rand(2,1);
        x2_sys = A2*x2_sys + Qv*randn(4,1);
        [~,chi2,~,P2] = informationKalman(chi2,P2,y2,A2,C,Qv,Qw);
        count2(ii) = count2(ii)+1;
        K2 = pinv(P2);
    end
    
end

if (mean(count1)<10)
    N1 = 10;
else
    N1 = ceil(mean(count1));
end

if (mean(count2)<10)
    N2 = 10;
else
    N2 = ceil(mean(count2));
end

%% Information Kalman Filter Results

num_trials = 5;
x_sys = zeros(4,N1+N2,num_trials);
P_pred = zeros(4,4,N1+N2,num_trials); P_est = zeros(4,4,N1+N2,num_trials);
chi_pred = zeros(4,N1+N2,num_trials); chi_est = zeros(4,N1+N2,num_trials);
x_pred = zeros(4,N1+N2,num_trials); x_est = zeros(4,N1+N2,num_trials);

for jj = 1:num_trials
    
    x_sys(:,1,jj) = randn(4,1); 
    P_pred(:,:,1,jj) = eye(4);
    chi_pred(:,1,jj) = x_sys(:,1,jj);
    x_pred(:,1,jj) = pinv(P_pred(:,:,1,jj))*chi_pred(:,1,jj);

    for ii = 1:(N1-1)
        x_sys(:,ii+1,jj) = A1*x_sys(:,ii,jj) + Qv*randn(4,1);
        y = C*x_sys(:,ii,jj) + Qw*randn(2,1);
        [chi_est(:,ii,jj),chi_pred(:,ii+1,jj),P_est(:,:,ii,jj),P_pred(:,:,ii+1,jj)] = ...
            informationKalman(chi_pred(:,ii,jj),P_pred(:,:,ii,jj),y,A1,C,Qv,Qw);
        x_est(:,ii,jj) = pinv(P_est(:,:,ii,jj))*chi_est(:,ii,jj);
        x_pred(:,ii+1,jj) = pinv(P_pred(:,:,ii+1,jj))*chi_pred(:,ii+1,jj);
    end
    
    for ii = N1:(N1+N2-1)
        x_sys(:,ii+1,jj) = A2*x_sys(:,ii,jj) + Qv*randn(4,1);
        y = C*x_sys(:,ii,jj) + Qw*randn(2,1);
        [chi_est(:,ii,jj),chi_pred(:,ii+1,jj),P_est(:,:,ii,jj),P_pred(:,:,ii+1,jj)] = ...
            informationKalman(chi_pred(:,ii,jj),P_pred(:,:,ii,jj),y,A2,C,Qv,Qw);
        x_est(:,ii,jj) = pinv(P_est(:,:,ii,jj))*chi_est(:,ii,jj);
        x_pred(:,ii+1,jj) = pinv(P_pred(:,:,ii+1,jj))*chi_pred(:,ii+1,jj);
    end
    
end

figure(5)
plot(1:N1+N2,reshape(vecnorm(x_pred),[N1+N2,num_trials]),'Linewidth',1);
legend_inf = legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5','location','best');
set(legend_inf,'Interpreter','latex');
title('Magnitude of State (Information Kalman Filter)','interpreter','latex')
xlabel('Time ($n$)','interpreter','latex'); 
ylabel('Magnitude ($|\chi [n]|$)','interpreter','latex'); 

figure(6)
plot3(reshape(x_pred(1,1:N1,1),[N1,1]),...
      reshape(x_pred(2,1:N1,1),[N1,1]),...
      reshape(x_pred(3,1:N1,1),[N1,1]),'blue','Linewidth',1);
hold on;
plot3(reshape(x_pred(1,N1:N1+N2,1),[N2+1,1]),...
      reshape(x_pred(2,N1:N1+N2,1),[N2+1,1]),...
      reshape(x_pred(3,N1:N1+N2,1),[N2+1,1]),'red','Linewidth',1);
legend_inf3d = legend('A1','A2','location','best');
set(legend_inf3d,'Interpreter','latex');
title('Three Components of State (Information Kalman Filter)','interpreter','latex')
xlabel('$x _1[n]$','interpreter','latex'); 
ylabel('$x _2[n]$','interpreter','latex'); 
zlabel('$x _3[n]$','interpreter','latex');

%% Information Kalman Filter Analysis

norm_deltax_pred = reshape(vecnorm(x_sys-x_pred),[N1+N2,num_trials]);
norm_deltax_est = reshape(vecnorm(x_sys-x_est),[N1+N2,num_trials]);

norm_deltaP_pred = zeros(N1+N2,1);
for jj = 1:N1
    norm_deltaP_pred(jj) = norm(P_pred(:,:,jj,1)-pinv(K1_ss));
end
for jj = N1+1:N2
    norm_deltaP_pred(jj) = norm(P_pred(:,:,jj,1)-pinv(K2_ss));
end

norm_P_pred = zeros(N1+N2,1);
for jj = 1:N1+N2
    norm_P_pred(jj) = norm(P_pred(:,:,jj,1));
end

norm_P_est = zeros(N1+N2,1);
for jj = 1:N1+N2
    norm_P_est(jj) = norm(P_est(:,:,jj,1));
end

norm_deltaP_est = zeros(N1+N2-1,1);
for jj = 1:N1+N2-1
    norm_deltaP_est(jj) = norm(P_est(:,:,jj+1,1)-P_est(:,:,jj,1));
end

f7 = figure(7);
f7.Name = "Information Kalman Filter";

subplot(2,3,1)
plot(1:N1+N2,norm_deltax_pred,'Linewidth',1);
legend_cov = legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5','location','best');
set(legend_cov,'Interpreter','latex');
title('$\| x(n) - \hat{x}(n|n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,2)
plot(1:N1+N2,norm_deltax_est,'Linewidth',1);
legend_cov = legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5','location','best');
set(legend_cov,'Interpreter','latex');
title('$\| x(n) - \hat{x}(n|n) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([1,N1+N2-1])

subplot(2,3,3)
plot(1:N1+N2,norm_deltaP_pred,'Linewidth',1);
title('$\| P(n,n-1) - P_{i,ss} \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,4)
plot(1:N1+N2,norm_P_pred,'Linewidth',1);
title('$\| P(n,n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,5)
plot(1:N1+N2,norm_P_est,'Linewidth',1);
title('$\| P(n,n) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,6)
plot(1:N1+N2-1,norm_deltaP_est,'Linewidth',1);
title('$\| P(n,n)-P(n-1,n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

%% Compare Covariance and Information Kalman Filters

N = 20;

x_sys = zeros(4,2*N); y = zeros(2,2*N);

x_estC = zeros(4,2*N); x_predC = zeros(4,2*N); 
x_estI = zeros(4,2*N); x_predI = zeros(4,2*N); 

K_estC = zeros(4,4,2*N); K_predC = zeros(4,4,2*N);
K_estI = zeros(4,4,2*N); K_predI = zeros(4,4,2*N);

x_sys(:,1) = randn(4,1); x_predC(:,1) = x_sys(:,1); x_predI(:,1) = x_sys(:,1);
K_predC(:,:,1) = 10000*eye(4); K_predI(:,:,1) = 10000*eye(4);

chi_est = zeros(4,2*N); chi_pred = zeros(4,2*N); 
P_est = zeros(4,4,2*N); P_pred = zeros(4,4,2*N);

P_pred(:,:,1) = pinv(K_estI(:,:,1)); chi_pred(:,1) = P_pred(:,:,1)*x_predI(:,1);

for ii = 1:N-1
    x_sys(:,ii+1) = A1*x_sys(:,ii) + Qv*randn(4,1);
    y(:,ii) = C*x_sys(:,ii) + Qw*randn(2,1);
    
    [x_estC(:,ii),x_predC(:,ii+1),K_estC(:,:,ii),K_predC(:,:,ii+1),~] = ...
        covarianceKalman(x_predC(:,ii),K_predC(:,:,ii),y(:,ii),A1,C,Qv,Qw);
    
    [chi_est(:,ii),chi_pred(:,ii+1),P_est(:,:,ii),P_pred(:,:,ii+1)] = ...
        informationKalman(chi_pred(:,ii),P_pred(:,:,ii),y(:,ii),A1,C,Qv,Qw);
    
    x_estI(:,ii) = pinv(P_est(:,:,ii))*chi_est(:,ii);
    x_predI(:,ii+1) = pinv(P_pred(:,:,ii+1))*chi_pred(:,ii+1);
end
for ii = N:2*N-1
    x_sys(:,ii+1) = A1*x_sys(:,ii) + Qv*randn(4,1);
    y(:,ii) = C*x_sys(:,ii) + Qw*randn(2,1);
    
    [x_estC(:,ii),x_predC(:,ii+1),K_estC(:,:,ii),K_predC(:,:,ii+1),~] = ...
        covarianceKalman(x_predC(:,ii),K_predC(:,:,ii),y(:,ii),A1,C,Qv,Qw);
    
    [chi_est(:,ii),chi_pred(:,ii+1),P_est(:,:,ii),P_pred(:,:,ii+1)] = ...
        informationKalman(chi_pred(:,ii),P_pred(:,:,ii),y(:,ii),A1,C,Qv,Qw);
    
    x_estI(:,ii) = pinv(P_est(:,:,ii))*chi_est(:,ii);
    x_predI(:,ii+1) = pinv(P_pred(:,:,ii+1))*chi_pred(:,ii+1);
end

%% Show Comparison

norm_deltax_predC = vecnorm(x_sys-x_predC);
norm_deltax_predI = vecnorm(x_sys-x_predI);

norm_deltax_estC = vecnorm(x_sys-x_estC);
norm_deltax_estI = vecnorm(x_sys-x_estI);

norm_deltaK_predC = zeros(2*N,1);
norm_deltaK_predI = zeros(2*N,1);
for ii = 1:N
    norm_deltaK_predC(ii) = norm(K_predC(:,:,ii)-K1_ss);
    norm_deltaK_predI(ii) = norm(K_predI(:,:,ii)-K1_ss);
end
for ii = N+1:2*N
    norm_deltaK_predC(ii) = norm(K_predC(:,:,ii)-K2_ss);
    norm_deltaK_predI(ii) = norm(K_predI(:,:,ii)-K2_ss);
end

norm_K_predC = zeros(2*N,1);
norm_K_predI = zeros(2*N,1);
for ii = 1:N1+N2
    norm_K_predC(ii) = norm(K_predC(:,:,ii));
    norm_K_predI(ii) = norm(K_predI(:,:,ii));
end

norm_K_estC = zeros(2*N,1);
norm_K_estI = zeros(2*N,1);
for ii = 1:2*N
    norm_K_estC(ii) = norm(K_estC(:,:,ii));
    norm_K_estI(ii) = norm(K_estI(:,:,ii));
end

norm_deltaK_estC = zeros(2*N-1,1);
norm_deltaK_estI = zeros(2*N-1,1);
for ii = 1:N1+N2-1
    norm_deltaK_estC(ii) = norm(K_estC(:,:,ii+1)-K_estC(:,:,ii));
    norm_deltaK_estI(ii) = norm(K_estI(:,:,ii+1)-K_estI(:,:,ii));
end

f8 = figure(8);
f8.Name = "Covariance and Information Kalman Filters";

subplot(2,3,1)
plot(1:2*N,norm_deltax_predC,1:2*N,norm_deltax_predI,'Linewidth',1);
legend_1 = legend('Covariance Kalman Filter','Information Kalman Filter','location','best');
set(legend_1,'Interpreter','latex');
title('$\| x(n) - \hat{x}(n|n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,2)
plot(1:2*N,norm_deltax_estC,1:2*N,norm_deltax_estI,'Linewidth',1);
legend_2 = legend('Covariance Kalman Filter','Information Kalman Filter','location','best');
set(legend_2,'Interpreter','latex');
title('$\| x(n) - \hat{x}(n|n) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([1,2*N-1]);

subplot(2,3,3)
plot(1:2*N,norm_deltaK_predC,1:2*N,norm_deltaK_predI,'Linewidth',1);
legend_3 = legend('Covariance Kalman Filter','Information Kalman Filter','location','best');
set(legend_3,'Interpreter','latex');
title('$\| K(n,n-1) - K_{i,ss} \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,4)
plot(1:2*N,norm_K_predC,1:2*N,norm_K_predI,'Linewidth',1);
legend_4 = legend('Covariance Kalman Filter','Information Kalman Filter','location','best');
set(legend_4,'Interpreter','latex');
title('$\| K(n,n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,5)
plot(1:2*N,norm_K_estC,1:2*N,norm_K_estI,'Linewidth',1);
legend_5 = legend('Covariance Kalman Filter','Information Kalman Filter','location','best');
set(legend_5,'Interpreter','latex');
title('$\| K(n,n) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,6)
plot(1:2*N-1,norm_deltaK_estC,1:2*N-1,norm_deltaK_estI,'Linewidth',1);
legend_6 = legend('Covariance Kalman Filter','Information Kalman Filter','location','best');
set(legend_6,'Interpreter','latex');
title('$\| K(n,n)-K(n-1,n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

%% Square Root Kalman Filter Setup

num_runs = 100;
count1 = zeros(num_runs,1); count2 = zeros(num_runs,1);

for jj = 1:num_runs
    x1 = randn(4,1); x1_sys = x1;
    K1 = eye(4);
    while (norm(K1-K1_ss)/norm(K1_ss) > 0.01)
        y1 = C*x1_sys + Qw*rand(2,1);
        x1_sys = A1*x1_sys + Qv*randn(4,1);
        [~,x1,~,K1,~] = squarerootKalman(x1,K1,y1,A1,C,Qv,Qw);
        count1(jj) = count1(jj)+1;
    end
    x2 = randn(4,1); x2_sys = x2;
    K2 = eye(4);
    while (norm(K2-K2_ss)/norm(K2_ss) > 0.01)
        y2 = C*x2_sys + Qw*rand(2,1);
        x2_sys = A2*x2_sys + Qv*randn(4,1);
        [~,x2,~,K2,~] = squarerootKalman(x2,K2,y2,A2,C,Qv,Qw);
        count2(jj) = count2(jj)+1;
    end
end

if (mean(count1)<10)
    N1 = 10;
else
    N1 = ceil(mean(count1));
end

if (mean(count2)<10)
    N2 = 10;
else
    N2 = ceil(mean(count2));
end

%% Square Root Kalman Filter Results

num_trials = 5;
x_sys = zeros(4,N1+N2,num_trials);
x_pred = zeros(4,N1+N2,num_trials);
K_pred = zeros(4,4,N1+N2,num_trials);
x_est = zeros(4,N1+N2,num_trials);
K_est = zeros(4,4,N1+N2,num_trials);

for jj = 1:num_trials
    x_sys(:,1,jj) = randn(4,1); 
    x_pred(:,1,jj) = x_sys(:,1,jj); 
    K_pred(:,:,1,jj) = eye(4);
    for ii = 1:(N1-1)
        x_sys(:,ii+1,jj) = A1*x_sys(:,ii,jj) + Qv*randn(4,1);
        y = C*x_sys(:,ii,jj) + Qw*randn(2,1);
        [x_est(:,ii,jj),x_pred(:,ii+1,jj),K_est(:,:,ii,jj),K_pred(:,:,ii+1,jj),G] = ...
            squarerootKalman(x_pred(:,ii,jj),K_pred(:,:,ii,jj),y,A1,C,Qv,Qw);
    end
    for ii = N1:(N1+N2-1)
        x_sys(:,ii+1,jj) = A2*x_sys(:,ii,jj) + Qv*randn(4,1);
        y = C*x_sys(:,ii,jj) + Qw*randn(2,1);
        [x_est(:,ii,jj),x_pred(:,ii+1,jj),K_est(:,:,ii,jj),K_pred(:,:,ii+1,jj),G] = ...
            squarerootKalman(x_pred(:,ii,jj),K_pred(:,:,ii,jj),y,A2,C,Qv,Qw);
    end
end

figure(9)
plot(1:N1+N2,reshape(vecnorm(x_pred),[N1+N2,num_trials]),'Linewidth',1);
legend_cov = legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5','location','best');
set(legend_cov,'Interpreter','latex');
title('Magnitude of State (Square Root Kalman Filter)','interpreter','latex')
xlabel('Time ($n$)','interpreter','latex'); 
ylabel('Magnitude ($|x[n]|$)','interpreter','latex'); 

figure(10)
plot3(reshape(x_pred(1,1:N1,1),[N1,1]),...
      reshape(x_pred(2,1:N1,1),[N1,1]),...
      reshape(x_pred(3,1:N1,1),[N1,1]),'blue','Linewidth',1);
hold on;
plot3(reshape(x_pred(1,N1:N1+N2,1),[N2+1,1]),...
      reshape(x_pred(2,N1:N1+N2,1),[N2+1,1]),...
      reshape(x_pred(3,N1:N1+N2,1),[N2+1,1]),'red','Linewidth',1);
legend_cov3d = legend('A1','A2','location','best');
set(legend_cov3d,'Interpreter','latex');
title('Three Components of State (Square Root Kalman Filter)','interpreter','latex')
xlabel('$x_1[n]$','interpreter','latex'); 
ylabel('$x_2[n]$','interpreter','latex'); 
zlabel('$x_3[n]$','interpreter','latex');

%% Square Root Kalman Filter Analysis

norm_deltax_pred = reshape(vecnorm(x_sys-x_pred),[N1+N2,num_trials]);
norm_deltax_est = reshape(vecnorm(x_sys-x_est),[N1+N2,num_trials]);

norm_deltaK_pred = zeros(N1+N2,1);
for jj = 1:N1
    norm_deltaK_pred(jj) = norm(K_pred(:,:,jj,1)-K1_ss);
end
for jj = N1+1:N2
    norm_deltaK_pred(jj) = norm(K_pred(:,:,jj,1)-K2_ss);
end

norm_K_pred = zeros(N1+N2,1);
for jj = 1:N1+N2
    norm_K_pred(jj) = norm(K_pred(:,:,jj,1));
end

norm_K_est = zeros(N1+N2,1);
for jj = 1:N1+N2
    norm_K_est(jj) = norm(K_est(:,:,jj,1));
end

norm_deltaK_est = zeros(N1+N2-1,1);
for jj = 1:N1+N2-1
    norm_deltaK_est(jj) = norm(K_est(:,:,jj+1,1)-K_est(:,:,jj,1));
end

f11 = figure(11);
f11.Name = "Square Root Kalman Filter";

subplot(2,3,1)
plot(1:N1+N2,norm_deltax_pred,'Linewidth',1);
legend_cov = legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5','location','best');
set(legend_cov,'Interpreter','latex');
title('$\| x(n) - \hat{x}(n|n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,2)
plot(1:N1+N2,norm_deltax_est,'Linewidth',1);
legend_cov = legend('Trial 1','Trial 2','Trial 3','Trial 4','Trial 5','location','best');
set(legend_cov,'Interpreter','latex');
title('$\| x(n) - \hat{x}(n|n) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([1,N1+N2-1])

subplot(2,3,3)
plot(1:N1+N2,norm_deltaK_pred,'Linewidth',1);
title('$\| K(n|n-1) - K_{i,ss} \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,4)
plot(1:N1+N2,norm_K_pred,'Linewidth',1);
title('$\| K(n|n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,5)
plot(1:N1+N2,norm_K_est,'Linewidth',1);
title('$\| K(n,n) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(2,3,6)
plot(1:N1+N2-1,norm_deltaK_est,'Linewidth',1);
title('$\| K(n,n)-K(n-1,n-1) \|$','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

%% Extended Kalman Filter - Case 1

N = 100;
Qv = 0.2*eye(4); Qw = 0.1*eye(4);

x_sys = zeros(4,N); x_est = zeros(4,N); x_pred = zeros(4,N); 
K_est = zeros(4,4,N); K_pred = zeros(4,4,N); y = zeros(4,N);

x_sys(:,1) = ones(4,1); x_pred(:,1) = ones(4,1); 
K_pred(:,:,1) = eye(4);

for ii = 1:N-1
    [f,~] = EKF_F(x_sys(:,ii));
    x_sys(:,ii+1) = f + Qv*randn(4,1);
    y(:,ii) = sin(x_sys(:,ii)) + Qw*randn(4,1);
    
    [x_est(:,ii),x_pred(:,ii+1),K_est(:,:,ii),K_pred(:,:,ii+1)] = ...
        extendedKalman(x_pred(:,ii),K_pred(:,:,ii),y(:,ii),Qv,Qw,1,1);
end

figure(12)

subplot(3,1,1)
plot(1:N,x_sys.','Linewidth',1);
legend_ekf1 = legend('$x_1$','$x_2$','$x_3$','$x_4$','location','best');
set(legend_ekf1,'Interpreter','latex');
title('Predicted State ($h(x)=\sin (x)$)','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(3,1,2)
plot(1:N,y.','Linewidth',1);
legend_ekf2 = legend('$y_1$','$y_2$','$y_3$','$y_4$','location','best');
set(legend_ekf2,'Interpreter','latex');
title('Measurements ($h(x)=\sin (x)$)','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(3,1,3)
plot(1:N,vecnorm(x_sys-x_est),'Linewidth',1);
title('$\|x(n) - \hat{x}(n|n) \|$ ($h(x)=\sin (x)$)','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

%% Extended Kalman Filter - Case 2

x_sys = zeros(4,N); x_est = zeros(4,N); x_pred = zeros(4,N); 
K_est = zeros(4,4,N); K_pred = zeros(4,4,N); y = zeros(4,N);

x_sys(:,1) = ones(4,1); x_pred(:,1) = ones(4,1); 
K_pred(:,:,1) = eye(4);

for ii = 1:N-1
    [f,~] = EKF_F(x_sys(:,ii));
    x_sys(:,ii+1) = f + Qv*randn(4,1);
    y(:,ii) = sin(x_sys(:,ii)) + Qw*randn(4,1);
    
    [x_est(:,ii),x_pred(:,ii+1),K_est(:,:,ii),K_pred(:,:,ii+1)] = ...
        extendedKalman(x_pred(:,ii),K_pred(:,:,ii),y(:,ii),Qv,Qw,1,2);
end

figure(13)

subplot(3,1,1)
plot(1:N,x_sys.','Linewidth',1);
legend_ekf1 = legend('$x_1$','$x_2$','$x_3$','$x_4$','location','best');
set(legend_ekf1,'Interpreter','latex');
title('Predicted State ($h(x)=\arctan (x)$)','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(3,1,2)
plot(1:N,y.','Linewidth',1);
legend_ekf2 = legend('$y_1$','$y_2$','$y_3$','$y_4$','location','best');
set(legend_ekf2,'Interpreter','latex');
title('Measurements ($h(x)=\arctan (x)$)','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(3,1,3)
plot(1:N,vecnorm(x_sys-x_est),'Linewidth',1);
title('$\|x(n) - \hat{x}(n|n) \|$ ($h(x)=\arctan (x)$)','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

%% Extended Kalman Filter - Case 3

x_sys = zeros(4,N); x_est = zeros(4,N); x_pred = zeros(4,N); 
K_est = zeros(4,4,N); K_pred = zeros(4,4,N); y = zeros(4,N);

x_sys(:,1) = ones(4,1); x_pred(:,1) = ones(4,1); 
K_pred(:,:,1) = eye(4);

for ii = 1:N-1
    [f,~] = EKF_F(x_sys(:,ii));
    x_sys(:,ii+1) = f + Qv*randn(4,1);
    y(:,ii) = sin(x_sys(:,ii)) + Qw*randn(4,1);
    
    [x_est(:,ii),x_pred(:,ii+1),K_est(:,:,ii),K_pred(:,:,ii+1)] = ...
        extendedKalman(x_pred(:,ii),K_pred(:,:,ii),y(:,ii),Qv,Qw,2,2);
end

figure(14)

subplot(3,1,1)
plot(1:N,x_sys.','Linewidth',1);
legend_ekf1 = legend('$x_1$','$x_2$','$x_3$','$x_4$','location','best');
set(legend_ekf1,'Interpreter','latex');
title('Predicted State (Nondifferentiable System)','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(3,1,2)
plot(1:N,y.','Linewidth',1);
legend_ekf2 = legend('$y_1$','$y_2$','$y_3$','$y_4$','location','best');
set(legend_ekf2,'Interpreter','latex');
title('Measurements (Nondifferentiable System)','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 

subplot(3,1,3)
plot(1:N,vecnorm(x_sys-x_est),'Linewidth',1);
title('$\|x(n) - \hat{x}(n|n) \|$ (Nondifferentiable System)','interpreter','latex')
xlabel('Time','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
