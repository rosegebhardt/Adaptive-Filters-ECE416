%% Rose Gebhardt: PSET3 Question 1
clear all; close all; clc; %#ok<CLALL>

%% Part A

poles1 = [0.3;0.5];
poles2 = [0.3;0.95];
noise_var = 1;

M = [3;6;10];
beta = (1./((1:6).^2)).';

a1 = [-(poles1(1)+poles1(2));poles1(1)*poles1(2)];
a2 = [-(poles2(1)+poles2(2));poles2(1)*poles2(2)];

s1 = sign(-poles1(1)/(1+poles1(1)^2) + poles1(2)/(1+poles1(2)^2));
s2 = sign(-poles2(1)/(1+poles2(1)^2) + poles2(2)/(1+poles2(2)^2));

beta1 = abs(-poles1(1)*(1+poles1(2)^2) + poles1(2)*(1+poles1(1)^2));
beta2 = abs(-poles2(1)*(1+poles2(2)^2) + poles2(2)*(1+poles2(1)^2));

c1 = noise_var^4/beta1;
c2 = noise_var^4/beta2;

rm1 = zeros(10,1);
rm2 = zeros(10,1);
for index = 1:10
    rm1(index) = c1*s1*((-poles1(1)/(1-poles1(1)^2))*poles1(1)^abs(index)+...
        (poles1(2)/(1-poles1(2)^2))*poles1(2)^abs(index));
    rm2(index) = c2*s2*((-poles2(1)/(1-poles2(1)^2))*poles2(1)^abs(index)+...
        (poles2(2)/(1-poles2(2)^2))*poles2(2)^abs(index));
end

R3_1 = toeplitz(rm1(1:3)); R6_1 = toeplitz(rm1(1:6)); R10_1 = toeplitz(rm1(1:10));
R3_2 = toeplitz(rm2(1:3)); R6_2 = toeplitz(rm2(1:6)); R10_2 = toeplitz(rm2(1:10));

p3_1 = R3_1*beta(1:3); p6_1 = R6_1*beta; p10_1 = R10_1*[beta;zeros(4,1)];
p3_2 = R3_2*beta(1:3); p6_2 = R6_2*beta; p10_2 = R10_2*[beta;zeros(4,1)];

w03_1 = pinv(R3_1)*p3_1; w06_1 = pinv(R6_1)*p6_1; w010_1 = pinv(R10_1)*p10_1;
w03_2 = pinv(R3_2)*p3_2; w06_2 = pinv(R6_2)*p6_2; w010_2 = pinv(R10_2)*p10_2;

delta3_1 = abs(beta(1:3)-w03_1); delta6_1 = abs(beta-w06_1); delta10_1 = abs([beta;zeros(4,1)]-w010_1);
delta3_2 = abs(beta(1:3)-w03_2); delta6_2 = abs(beta-w06_2); delta10_2 = abs([beta;zeros(4,1)]-w010_2);

evals3_1 = eig(R3_1); evals6_1 = eig(R6_1); evals10_1 = eig(R10_1);
evals3_2 = eig(R3_2); evals6_2 = eig(R6_2); evals10_2 = eig(R10_2);

spread3_1 = max(evals3_1) - min(evals3_1);
spread6_1 = max(evals6_1) - min(evals6_1);
spread10_1 = max(evals10_1) - min(evals10_1);
spread3_2 = max(evals3_2) - min(evals3_2);
spread6_2 = max(evals6_2) - min(evals6_2);
spread10_2 = max(evals10_2) - min(evals10_2);
% The second case has a much larger eigenvalue spread

mu_max3_1 = 2/max(evals3_1); mu_max6_1 = 2/max(evals6_1); mu_max10_1 = 2/max(evals10_1);
mu_max3_2 = 2/max(evals3_2); mu_max6_2 = 2/max(evals6_2); mu_max10_2 = 2/max(evals10_2);

mu_tilde_max = 2;

%% Part B

num_runs = 100;

%% (Case 1, M = 3)

N = 400;

u3_1 = form_U(a1,6,N);
[D3_1_05,J3_1_05] = learning_curve(0.05*mu_max3_1,[w03_1;zeros(3,1)],beta,u3_1,N,num_runs,'LMS');
[D3_1_50,J3_1_50] = learning_curve(0.50*mu_max3_1,[w03_1;zeros(3,1)],beta,u3_1,N,num_runs,'LMS');
[D3_1_80,J3_1_80] = learning_curve(0.80*mu_max3_1,[w03_1;zeros(3,1)],beta,u3_1,N,num_runs,'LMS');
[D3_1_20,J3_1_20] = learning_curve(0.20*mu_tilde_max,[w03_1;zeros(3,1)],beta,u3_1,N,num_runs,'NLMS');

figure(1)

subplot(4,2,1)
plot(J3_1_05,'LineWidth',1)
title('Learning Curve (Case 1, M=3, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,2)
plot(D3_1_05,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=3, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,3)
plot(J3_1_50,'LineWidth',1)
title('Learning Curve (Case 1, M=3, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,4)
plot(D3_1_50,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=3, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,5)
plot(J3_1_80,'LineWidth',1)
title('Learning Curve (Case 1, M=3, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,6)
plot(D3_1_80,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=3, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,7)
plot(J3_1_20,'LineWidth',1)
title('Learning Curve (Case 1, M=3, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,8)
plot(D3_1_20,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=3, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

%% (Case 1, M = 6)

N = 400;

u6_1 = form_U(a1,6,N);
[D6_1_05,J6_1_05] = learning_curve(0.05*mu_max6_1,w06_1,beta,u6_1,N,num_runs,'LMS');
[D6_1_50,J6_1_50] = learning_curve(0.50*mu_max6_1,w06_1,beta,u6_1,N,num_runs,'LMS');
[D6_1_80,J6_1_80] = learning_curve(0.80*mu_max6_1,w06_1,beta,u6_1,N,num_runs,'LMS');
[D6_1_20,J6_1_20] = learning_curve(0.20*mu_tilde_max,w06_1,beta,u6_1,N,num_runs,'NLMS');

figure(2)

subplot(4,2,1)
plot(J6_1_05,'LineWidth',1)
title('Learning Curve (Case 1, M=6, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,2)
plot(D6_1_05,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=6, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,3)
plot(J6_1_50,'LineWidth',1)
title('Learning Curve (Case 1, M=6, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,4)
plot(D6_1_50,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=6, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,5)
plot(J6_1_80,'LineWidth',1)
title('Learning Curve (Case 1, M=6, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,6)
plot(D6_1_80,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=6, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 
    
subplot(4,2,7)
plot(J6_1_20,'LineWidth',1)
title('Learning Curve (Case 1, M=6, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,8)
plot(D6_1_20,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=6, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

%% (Case 1, M = 10)

N = 400;

u10_1 = form_U(a1,10,N);
[D10_1_05,J10_1_05] = learning_curve(0.05*mu_max10_1,w010_1,[beta;zeros(4,1)],u10_1,N,num_runs,'LMS');
[D10_1_50,J10_1_50] = learning_curve(0.50*mu_max10_1,w010_1,[beta;zeros(4,1)],u10_1,N,num_runs,'LMS');
[D10_1_80,J10_1_80] = learning_curve(0.80*mu_max10_1,w010_1,[beta;zeros(4,1)],u10_1,N,num_runs,'LMS');
[D10_1_20,J10_1_20] = learning_curve(0.20*mu_tilde_max,w010_1,[beta;zeros(4,1)],u10_1,N,num_runs,'NLMS');

figure(3)

subplot(4,2,1)
plot(J10_1_05,'LineWidth',1)
title('Learning Curve (Case 1, M=10, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,2)
plot(D10_1_05,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=10, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,3)
plot(J10_1_50,'LineWidth',1)
title('Learning Curve (Case 1, M=10, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,4)
plot(D10_1_50,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=10, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,5)
plot(J10_1_80,'LineWidth',1)
title('Learning Curve (Case 1, M=10, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,6)
plot(D10_1_80,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=10, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 
    
subplot(4,2,7)
plot(J10_1_20,'LineWidth',1)
title('Learning Curve (Case 1, M=10, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,8)
plot(D10_1_20,'LineWidth',1)
title('MSD Learning Curve (Case 1, M=10, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

%% (Case 2, M = 3)

N = 500;

u3_2 = form_U(a2,6,N);
[D3_2_05,J3_2_05] = learning_curve(0.05*mu_max3_2,[w03_2;zeros(3,1)],beta,u3_2,N,num_runs,'LMS');
[D3_2_50,J3_2_50] = learning_curve(0.50*mu_max3_2,[w03_2;zeros(3,1)],beta,u3_2,N,num_runs,'LMS');
[D3_2_80,J3_2_80] = learning_curve(0.80*mu_max3_2,[w03_2;zeros(3,1)],beta,u3_2,N,num_runs,'LMS');
[D3_2_20,J3_2_20] = learning_curve(0.20*mu_tilde_max,[w03_2;zeros(3,1)],beta,u3_2,N,num_runs,'NLMS');

figure(4)

subplot(4,2,1)
plot(J3_2_05,'LineWidth',1)
title('Learning Curve (Case 2, M=3, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,2)
plot(D3_2_05,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=3, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,3)
plot(J3_2_50,'LineWidth',1)
title('Learning Curve (Case 2, M=3, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,4)
plot(D3_2_50,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=3, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,5)
plot(J3_2_80,'LineWidth',1)
title('Learning Curve (Case 2, M=3, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,6)
plot(D3_2_80,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=3, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,7)
plot(J3_2_20,'LineWidth',1)
title('Learning Curve (Case 2, M=3, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,8)
plot(D3_2_20,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=3, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

%% (Case 2, M = 6)

N = 500;

u6_2 = form_U(a2,6,N);
[D6_2_05,J6_2_05] = learning_curve(0.05*mu_max6_2,w06_2,beta,u6_2,N,num_runs,'LMS');
[D6_2_50,J6_2_50] = learning_curve(0.50*mu_max6_2,w06_2,beta,u6_2,N,num_runs,'LMS');
[D6_2_80,J6_2_80] = learning_curve(0.80*mu_max6_2,w06_2,beta,u6_2,N,num_runs,'LMS');
[D6_2_20,J6_2_20] = learning_curve(0.20*mu_tilde_max,w06_2,beta,u6_2,N,num_runs,'NLMS');

figure(5)

subplot(4,2,1)
plot(J6_2_05,'LineWidth',1)
title('Learning Curve (Case 2, M=6, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,2)
plot(D6_2_05,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=6, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,3)
plot(J6_2_50,'LineWidth',1)
title('Learning Curve (Case 2, M=6, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,4)
plot(D6_2_50,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=6, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,5)
plot(J6_2_80,'LineWidth',1)
title('Learning Curve (Case 2, M=6, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,6)
plot(D6_2_80,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=6, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 
    
subplot(4,2,7)
plot(J6_2_20,'LineWidth',1)
title('Learning Curve (Case 2, M=6, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,8)
plot(D6_2_20,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=6, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

%% (Case 1, M = 10)

N = 500;

u10_2 = form_U(a2,10,N);
[D10_2_05,J10_2_05] = learning_curve(0.05*mu_max10_2,w010_2,[beta;zeros(4,1)],u10_2,N,num_runs,'LMS');
[D10_2_50,J10_2_50] = learning_curve(0.50*mu_max10_2,w010_2,[beta;zeros(4,1)],u10_2,N,num_runs,'LMS');
[D10_2_80,J10_2_80] = learning_curve(0.80*mu_max10_2,w010_2,[beta;zeros(4,1)],u10_2,N,num_runs,'LMS');
[D10_2_20,J10_2_20] = learning_curve(0.20*mu_tilde_max,w010_2,[beta;zeros(4,1)],u10_2,N,num_runs,'NLMS');

figure(6)

subplot(4,2,1)
plot(J10_2_05,'LineWidth',1)
title('Learning Curve (Case 2, M=10, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,2)
plot(D10_2_05,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=10, $\mu=0.05\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,3)
plot(J10_2_50,'LineWidth',1)
title('Learning Curve (Case 2, M=10, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,4)
plot(D10_2_50,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=10, $\mu=0.50\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

subplot(4,2,5)
plot(J10_2_80,'LineWidth',1)
title('Learning Curve (Case 2, M=10, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,6)
plot(D10_2_80,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=10, $\mu=0.80\mu_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 
    
subplot(4,2,7)
plot(J10_2_20,'LineWidth',1)
title('Learning Curve (Case 2, M=10, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('J[n]','interpreter','latex'); 
subplot(4,2,8)
plot(D10_2_20,'LineWidth',1)
title('MSD Learning Curve (Case 2, M=10, $\mu=0.20\tilde{\mu}_{max}$)','interpreter','latex')
xlabel('Iterations','interpreter','latex'); 
ylabel('D[n]','interpreter','latex'); 

%% Part C

% Comments - stability, rate of convergence versus misadjustment, effect of
% eigenvalue spread, effect of model order, relation between J(n) and D(n)

% Using mu < mu_max did not always lead to stability. For most of the
% cases, only mu = 0.05mu_max (for LMS) and mu = 0.20mu_tilde_max (NLMS)
% were stable. For case 2, M = 10, mu = 0.5mu_max was also stable. This
% happens because the small-step size theory is not valid for large mu.

% As the parameter mu increased, the rate of convergence and the
% misadjustment both increased (had to try other values of mu to see this:
% most of these cases only had one stable case). This was expected.

% Case 2, which had a larger eigenvalue spread, converged slower than case
% 1. The smallest eigenvalue tended to be around the same, but the largest
% eigenvalue was much larger for the second case, making mu_max much
% smaller, so this trend is expected.

% The higher order models took longer to converge than the lower order
% models. This may be because the higher order models were able to fully
% describe the system while the lower order models were not.

% J(n) and D(n) behaved similarly, but D(n) tended to be a lot 'spikier' than
% J(n). As it converged, D(n) smoothed out. J(n) was defined based on e(n),
% which only uses w(n), not w0 whereas D(n) was based on w0-w(n). This was
% probably the cause of the spikiness. 
