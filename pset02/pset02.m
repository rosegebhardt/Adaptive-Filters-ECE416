%% Rose Gebhardt: PSET2 
clear all; close all; clc;

%% Question 2

% Part A

p_1 = 0.6;
p_2 = 0.8;
noise_var = 1;

a_1 = -(p_1+p_2);
a_2 = p_1*p_2;

S = @(w) noise_var^2 * abs(w^2/(a_2 + a_1*w + w^2));

s = sign(-p_1/(1+p_1^2) + p_2/(1+p_2^2));
beta = abs(-p_1*(1+p_2^2) + p_2*(1+p_1^2));
c = noise_var^4/beta;

rm = zeros(10,1);
for index = 1:10
    rm(index) = c*s*((-p_1/(1-p_1^2))*p_1^abs(index)...
        + (p_2/(1-p_2^2))*p_2^abs(index));
end
    
% Part B

N = 10^3;
v = sqrt(noise_var)*randn(1,N);
x = zeros(1,N);
x(1) = v(1);
x(2) = v(2) - a_1*x(1);
for index = 3:N
    x(index) = v(index) - a_1*x(index-1) - a_2*x(index-2);
end

rm_hat = zeros(10,1);
for index = 1:10
    rm_hat(index) = (x(index:end)*x(1:end-index+1)')/(N-index+1);
end

% Part C

Rm = toeplitz(rm);
Rm_hat = toeplitz(rm_hat);

max_abs_error = max(abs(rm-rm_hat));
spectral_norm_error = norm(Rm-Rm_hat);

% Part D

[K,P,A] = pset02_LD(rm.',9);
[K_hat,P_hat,A_hat] = pset02_LD(rm_hat.',9);

max_K_error = max(abs(K-K_hat));

% Part E

max_a1a2_error = max(abs(A(2:3,3)-A_hat(2:3,3)));
max_pole_error = max(abs([a_1;a_2]-A(2:3,3)));

% Part F

% This is true.

% Part G

figure(1)
subplot(1,2,1)
stem(P)
title('Prediction Error Powers for $$R_m$$','interpreter','latex')
xlabel('Order','interpreter','latex'); 
ylabel('Power','interpreter','latex'); 
subplot(1,2,2)
stem(P_hat)
title('Prediction Error Powers for $$\hat{R_m}$$','interpreter','latex')
xlabel('Order','interpreter','latex'); 
ylabel('Power','interpreter','latex'); 

% Part H

syms w
integral = int(log(S(w)),-pi,pi)/(2*pi);

% Part I

partI_vals = zeros(10,1);
for index = 1:10
    partI_vals(index) = log(det(toeplitz(rm(1:index))))/index;
end

figure(2)
stem(partI_vals)
title('$$\frac{1}{M} \log (\det R_m)$$','interpreter','latex')
xlabel('$$M$$','interpreter','latex'); 
ylabel('Value','interpreter','latex'); 

% Strictly decreasing, converge to 1

%% Question 3

% Part A

noise_var = 1;
N = 1000;
v = sqrt(noise_var)*randn(1,N);
u = zeros(1,N);

u(1) = v(1);
for index = 2:N
    u(index) = -0.5*u(index-1) + v(index) - 0.2*v(index-1);
end

rm_hat = zeros(4,1);
for index = 1:4
    rm_hat(index) = (u(index:end)*u(1:end-index+1)')/(N-index+1);
end

Rm_hat = toeplitz(rm_hat);
[~,D] = eig(Rm_hat);
lam_max = max(diag(D));
mu_max = 2/lam_max;

% Part B

[K_hat,P_hat,A_hat] = pset02_LD(rm_hat.',3);
a_fpef = A_hat(:,end);

% Part C

mu_01 = 0.1*mu_max;
w_old = zeros(4,1);
it01 = 0;
while max(abs(w_old-a_fpef))>1e-3
    w_new = (eye(4) - mu_01*Rm_hat)*w_old + mu_01*Rm_hat*a_fpef;
    w_old = w_new;
    it01 = it01 + 1;
end

mu_05 = 0.5*mu_max;
w_old = zeros(4,1);
it05 = 0;
while max(abs(w_old-a_fpef))>1e-3
    w_new = (eye(4) - mu_05*Rm_hat)*w_old + mu_05*Rm_hat*a_fpef;
    w_old = w_new;
    it05 = it05 + 1;
end

mu_09 = 0.9*mu_max;
w_old = zeros(4,1);
it09 = 0;
while max(abs(w_old-a_fpef))>1e-3
    w_new = (eye(4) - mu_09*Rm_hat)*w_old + mu_09*Rm_hat*a_fpef;
    w_old = w_new;
    it09 = it09 + 1;
end

% As expected it01 > it05 > it09 (373 > 73 > 39)

% Part D

rm = zeros(4,1);
Delta = zeros(4,1);
P = zeros(4,1);
A = zeros(4,4);

a = 0.5; b = -0.2;
A(1,:) = ones(1,4);
for index = 1:3
    A(index+1,4) = a*abs(b)^(index-1) + abs(b)^index;
end

for index = 4:-1:2
    for jndex = 2:index-1
        A(jndex,index-1) = (A(jndex,index)-A(index,index)*A(index-jndex,index)')...
        /(1-abs(A(index,index))^2);
    end
end
K = diag(A);

P(1) = rm_hat(1);
rm(1) = P(1);
for index = 1:3
    P(index+1) = P(index)*(1-abs(K(index+1))^2);
    Delta(index) = K(index+1)*P(index);
    rm(index+1) = (Delta(index) - flipud(rm(1:index))'*A(1:index,index))';
end

% rm is not the same as rm_hat

%% Question 4

% Part A

d_lam = 0.5;
M = 20;
Theta = deg2rad([10,30,50]);
r0z = [0;0;1]; m = 0:M-1; r = (m.*r0z).';

beta = [1;(10^(-5/10));(10^(-15/10))];
noise_var = 10^(-25/10);

S = beamformer(Theta,r,beta,d_lam,noise_var);
R = S*diag(beta)*S' + noise_var*eye(length(S));

% Part B

C_a = null(S');
g = eye(3);

w_mvdr = zeros(M,3);
w_q = zeros(M,3);
w_aopt = zeros(size(C_a,2),3);
w_gsc = zeros(M,3);

for index = 1:3
    w_mvdr(:,index) = (pinv(R)*S(:,index))/(S(:,index)'*pinv(R)*S(:,index));
    w_q(:,index) = S*pinv(S'*S)*g(:,index);
    w_aopt(:,index) = pinv(C_a'*R*C_a)*C_a'*R*w_q(:,index);
    w_gsc(:,index) = w_q(:,index)-C_a*w_aopt(:,index);
end

theta_range = linspace(0,pi,1000);
S_range = beamformer(theta_range,r,beta,d_lam,noise_var);
array_mvdr = zeros(1000,3);
array_gsc = zeros(1000,3);

for index = 1:3
    array_mvdr(:,index) = abs(S_range.'*w_mvdr(:,index)).^2;
    array_gsc(:,index) = abs(S_range.'*w_gsc(:,index)).^2;
end

figure(3)
subplot(1,2,1)
plot(rad2deg(theta_range),flip(array_mvdr(:,1)),'LineWidth',1)
title('MVDR Array Pattern $\theta = 10^{\mathrm o}$','interpreter','latex')
xlabel('$\theta$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([0,90])
subplot(1,2,2)
plot(rad2deg(theta_range),flip(array_gsc(:,1)),'LineWidth',1)
title('GSC Array Pattern $\theta = 10^{\mathrm o}$','interpreter','latex')
xlabel('$\theta$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([0,90])

figure(4)
subplot(1,2,1)
plot(rad2deg(theta_range),flip(array_mvdr(:,2)),'LineWidth',1)
title('MVDR Array Pattern $\theta = 30^{\mathrm o}$','interpreter','latex')
xlabel('$\theta$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([0,90])
subplot(1,2,2)
plot(rad2deg(theta_range),flip(array_gsc(:,2)),'LineWidth',1)
title('GSC Array Pattern $\theta = 30^{\mathrm o}$','interpreter','latex')
xlabel('$\theta$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([0,90])

figure(5)
subplot(1,2,1)
plot(rad2deg(theta_range),flip(array_mvdr(:,3)),'LineWidth',1)
title('MVDR Array Pattern $\theta = 50^{\mathrm o}$','interpreter','latex')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([0,90])
subplot(1,2,2)
plot(rad2deg(theta_range),flip(array_gsc(:,3)),'LineWidth',1)
title('GSC Array Pattern $\theta = 50^{\mathrm o}$','interpreter','latex')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([0,90])

[~,index_10] = min(abs(theta_range-Theta(1)));
[~,index_30] = min(abs(theta_range-Theta(2)));
[~,index_50] = min(abs(theta_range-Theta(3)));

loc = [index_10,index_30,index_50];

attenuation = zeros(3,3);
for index = 1:3
    for jndex = 1:3
        attenuation(index,jndex) = 10*log10(array_mvdr(end-loc(index),jndex));
    end
end
% Diagonal elements near zero, off-diagonal are large negative numbers
% Not attenuating source of interest, attenuates other sources well

% The differences between mainlobe and sidelobes of GSC and MVDR are very
% small - to small to see in plots

% Part D

d_lam = 0.4;
omega = linspace(-pi,pi,1000);
S_range = (1/sqrt(M))*exp(-1*1j*omega.*(0:M-1).');

array_mvdr = zeros(1000,3);
array_gsc = zeros(1000,3);

for index = 1:3
    array_mvdr(:,index) = abs(S_range.'*w_mvdr(:,index)).^2;
    array_gsc(:,index) = abs(S_range.'*w_gsc(:,index)).^2;
end

vis = rad2deg(2*pi*d_lam);

figure(6)
subplot(1,2,1)
plot(rad2deg(omega),flip(array_mvdr(:,1)),'LineWidth',1)
line([-vis,-vis], [0,1.1],'LineWidth',1); line([vis,vis], [0,1.1],'LineWidth',1);
title('MVDR Array Pattern $\theta = 10^{\mathrm o}$','interpreter','latex')
xlabel('$$\omega$$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([-180,180]); ylim([0,1.1])
subplot(1,2,2)
plot(rad2deg(omega),flip(array_gsc(:,1)),'LineWidth',1)
line([-vis,-vis], [0,1.1],'LineWidth',1); line([vis,vis], [0,1.1],'LineWidth',1);
title('GSC Array Pattern $\theta = 10^{\mathrm o}$','interpreter','latex')
xlabel('$$\omega$$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([-180,180]); ylim([0,1.1]);

figure(7)
subplot(1,2,1)
plot(rad2deg(omega),flip(array_mvdr(:,2)),'LineWidth',1)
line([-vis,-vis], [0,1.1],'LineWidth',1); line([vis,vis], [0,1.1],'LineWidth',1);
title('MVDR Array Pattern $\theta = 30^{\mathrm o}$','interpreter','latex')
xlabel('$$\omega$$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([-180,180]); ylim([0,1.1])
subplot(1,2,2)
plot(rad2deg(omega),flip(array_gsc(:,2)),'LineWidth',1)
line([-vis,-vis], [0,1.1],'LineWidth',1); line([vis,vis], [0,1.1],'LineWidth',1);
title('GSC Array Pattern $\theta =30^{\mathrm o}$','interpreter','latex')
xlabel('$$\omega$$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([-180,180]); ylim([0,1.1]);

figure(8)
subplot(1,2,1)
plot(rad2deg(omega),flip(array_mvdr(:,3)),'LineWidth',1)
line([-vis,-vis], [0,1.1],'LineWidth',1); line([vis,vis], [0,1.1],'LineWidth',1);
title('MVDR Array Pattern $\theta = 50^{\mathrm o}$','interpreter','latex')
xlabel('$$\omega$$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex'); 
xlim([-180,180]); ylim([0,1.1])
subplot(1,2,2)
plot(rad2deg(omega),flip(array_gsc(:,3)),'LineWidth',1)
line([-vis,-vis], [0,1.1],'LineWidth',1); line([vis,vis], [0,1.1],'LineWidth',1);
title('GSC Array Pattern $\theta = 50^{\mathrm o}$','interpreter','latex')
xlabel('$$\omega$$','interpreter','latex'); 
ylabel('Magnitude','interpreter','latex');
xlim([-180,180]); ylim([0,1.1]);

visible = (-2*pi*d_lam < omega) & (omega < 2*pi*d_lam);
invisible = ones(1,1000)-visible;

peak_invisible = zeros(3,2);

for index = 1:3
    peak_invisible(index,1) = max(invisible.'.*array_mvdr(:,index));
    peak_invisible(index,2) = max(invisible.'.*array_gsc(:,index));
end
