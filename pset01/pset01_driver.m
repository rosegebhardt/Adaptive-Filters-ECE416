%% Rose Gebhardt: PSET1
clear all; close all; clc;

%% Set Up

N = 100;
d_lam = 0.5;
m = -10:10;

theta = deg2rad([10,20,30]); phi = deg2rad([20,-20,150]);
Theta = [theta;phi];

r0x = [1;0;0]; r0y = [0;1;0];
rx = m.*r0x; ry = m.*r0y; r0 = [rx,ry];
[r,~,~] = unique(r0.','rows');

beta = repmat([1;(10^(-5/10));(10^(-10/10))],[1,N]);
noise_var = 10^(-20/10);

[S,A] = pset01a(Theta,r,beta,d_lam,noise_var);

R = S*diag(beta(:,1))*S' + noise_var*eye(length(S));
R_hat = pset01b(A);

%% Part 1: Eigenanalysis

[V,D] = eig(R);
sing_vals = sqrt(diag(D));

figure(1)
stem(real(sing_vals))
title('Theoretical Singular Values of A')

P_s = V(:,1:3)*V(:,1:3)';
P_n = eye(length(P_s)) - P_s;

part_1_works = abs(P_n*S) < 1e-12;

%% Part 2: Without White Noise

[~,A_0] = pset01a(Theta,r,beta,d_lam,0);
R_0 = S*diag(beta(:,1))*S';
    
[V_0,D_0] = eig(R_0);
sing_vals_0 = sqrt(diag(D_0));

figure(2)
stem(real([sing_vals,sing_vals_0]))
title('Comparison of R_0 and R')

% Singular values of the first three are about the same,
% but the singular values of the rest go to zero.

% The three eigenvectors corresponding to the largest eigenvalues are
% about the same, but the rest are different because of the noise added

%% Part 3: SVD

[u_A,s_A,v_A] = svd(A);

figure(3)
stem(diag(s_A));
title('Singular Values of A')

% Doesn't work...

%% Part 4: Signal-to-Noise Ratio

P_signals = sum(diag(s_A(1:3,1:3)^2));
P_noise = s_A(4,4)^2;

SNR = P_signals/P_noise;
SNR_ideal = sum(beta(:,1))/noise_var;

%% Part 5: Difference in R Matrices

norm_R_R_hat = norm(R-R_hat);

% Should be true, is not
part_5_works = norm_R_R_hat < D(4,4);

%% Part 6: MVDR and MUSIC Spectrums

gridsize = 100;
theta_domain = linspace(0,deg2rad(2),gridsize);
phi_domain = linspace(0,deg2rad(2),gridsize);
[theta,phi] = meshgrid(theta_domain,phi_domain);
Theta_Phi = [theta(:).';phi(:).'];

mvdr = zeros(length(Theta_Phi),1);
music = zeros(length(Theta_Phi),1);

for index = 1:length(Theta_Phi)
    [S_index,~] = pset01a(Theta_Phi(:,index),r,ones(1,N),d_lam,noise_var);
    s_index = S_index/norm(S_index);
    mvdr(index) = real(1/(s_index'*pinv(R_hat)*s_index));
    music(index) = real(1/(s_index'*P_n*s_index));
end

MVDR = reshape(mvdr,length(theta),length(phi));
MUSIC = reshape(music,length(theta),length(phi));

figure(4)
subplot(2,1,1)
contour(theta,phi,MVDR)
title('MVDR Spectrum')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MVDR}$$','interpreter','latex');
colorbar
subplot(2,1,2)
surf(theta,phi,MVDR,'EdgeColor','interp')
title('MVDR Spectrum')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MVDR}$$','interpreter','latex');
colorbar

figure(5)
subplot(2,1,1)
contour(theta,phi,MUSIC)
title('MUSIC Spectrum')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MUSIC}$$','interpreter','latex');
colorbar
subplot(2,1,2)
surf(theta,phi,MUSIC,'EdgeColor','interp')
title('MUSIC Spectrum')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MUSIC}$$','interpreter','latex');
colorbar

%% Part 7: Lower Bounds

min_MVDR = 1/norm(pinv(R_hat));
min_MUSIC = 1/norm(P_n);

%% Part 8: Source Vectors

MVDR_source = zeros(size(Theta,2),1);
MUSIC_source = zeros(size(Theta,2),1);

for index = 1:size(Theta,2)
    s_index = S(:,index)/norm(S(:,index));
    MVDR_source(index) = real(1/(s_index'*pinv(R_hat)*s_index));
    MUSIC_source(index) = real(1/(s_index'*P_n*s_index));
end

MVDR_peak_to_min = MVDR_source/min_MVDR;
MUSIC_peak_to_min = MUSIC_source/min_MUSIC;

%% Part 9: Grid Minimum

min_grid_MVDR = min(min(MVDR));
min_grid_MUSIC = min(min(MUSIC));

delta_min_MVDR = min_grid_MVDR-min_MVDR;
delta_min_MUSIC = min_grid_MUSIC-min_MUSIC;

%% Part 10a: Reduce Number of Samples (N=25)

% Set up
N = 25;

[S_25,A_25] = pset01a(Theta,r(11:31,:),beta,d_lam,noise_var);

R_25 = S_25*diag(beta(:,1))*S_25' + noise_var*eye(length(S_25));
R_hat_25 = pset01b(A_25);

% Part 1
[V_25,D_25] = eig(R_25);
sing_vals_25 = sqrt(diag(D_25));

figure(6)
stem(real(sing_vals_25))
title('Theoretical Singular Values of A (25 Samples)')

% Part 3
[u_A_25,s_A_25,v_A_25] = svd(A_25);

figure(7)
stem(diag(s_A_25));
title('Singular Values of A (25 Samples)')

% Part 4
P_signals_25 = sum(diag(s_A_25(1:3,1:3)^2));
P_noise_25 = s_A_25(4,4)^2;

SNR_25 = P_signals_25/P_noise_25;
SNR_ideal_25 = sum(beta(:,1))/noise_var;

% Part 5
norm_R_R_hat_25 = norm(R_25-R_hat_25);
part_5_works_25 = norm_R_R_hat_25 < D(4,4);

P_s_25 = V_25(:,1:3)*V_25(:,1:3)';
P_n_25 = eye(length(P_s_25)) - P_s_25;

% Part 6
mvdr_25 = zeros(length(Theta_Phi),1);
music_25 = zeros(length(Theta_Phi),1);

for index = 1:length(Theta_Phi)
    [S_index,~] = pset01a(Theta_Phi(:,index),r(11:31,:),ones(1,N),d_lam,noise_var);
    s_index = S_index/norm(S_index);
    mvdr_25(index) = real(1/(s_index'*pinv(R_hat_25)*s_index));
    music_25(index) = real(1/(s_index'*P_n_25*s_index));
end

MVDR_25 = reshape(mvdr_25,length(theta),length(phi));
MUSIC_25 = reshape(music_25,length(theta),length(phi));

figure(8)
subplot(2,1,1)
contour(theta,phi,MVDR_25)
title('MVDR Spectrum (N=25)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MVDR}$$','interpreter','latex');
colorbar
subplot(2,1,2)
surf(theta,phi,MVDR_25,'EdgeColor','interp')
title('MVDR Spectrum (N=25)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MVDR}$$','interpreter','latex');
colorbar

figure(9)
subplot(2,1,1)
contour(theta,phi,MUSIC_25)
title('MUSIC Spectrum (N=25)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MUSIC}$$','interpreter','latex');
colorbar
subplot(2,1,2)
surf(theta,phi,MUSIC_25,'EdgeColor','interp')
title('MUSIC Spectrum (N=25)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MUSIC}$$','interpreter','latex');
colorbar

% Part 7
min_MVDR_25 = 1/norm(pinv(R_hat_25));
min_MUSIC_25 = 1/norm(P_n_25);

% Part 8
MVDR_source_25 = zeros(size(Theta,2),1);
MUSIC_source_25 = zeros(size(Theta,2),1);

for index = 1:size(Theta,2)
    s_index = S_25(:,index)/norm(S_25(:,index));
    MVDR_source_25(index) = real(1/(s_index'*pinv(R_hat_25)*s_index));
    MUSIC_source_25(index) = real(1/(s_index'*P_n_25*s_index));
end

MVDR_peak_to_min_25 = MVDR_source_25/min_MVDR_25;
MUSIC_peak_to_min_25 = MUSIC_source_25/min_MUSIC_25;

% Part 9
min_grid_MVDR_25 = min(min(MVDR_25));
min_grid_MUSIC_25 = min(min(MUSIC_25));

delta_min_MVDR_25 = min_grid_MVDR_25-min_MVDR_25;
delta_min_MUSIC_25 = min_grid_MUSIC_25-min_MUSIC_25;

%% Part 10b: Reduce Number of Samples (N=10)

% Set up
N = 10;

[S_10,A_10] = pset01a(Theta,r(11:31,:),beta,d_lam,noise_var);

R_10 = S_10*diag(beta(:,1))*S_10' + noise_var*eye(length(S_10));
R_hat_10 = pset01b(A_10);

% Part 1
[V_10,D_10] = eig(R_10);
sing_vals_10 = sqrt(diag(D_10));

figure(10)
stem(real(sing_vals_10))
title('Theoretical Singular Values of A (10 Samples)')

% Part 3
[u_A_10,s_A_10,v_A_10] = svd(A_10);

figure(11)
stem(diag(s_A_10));
title('Singular Values of A (10 Samples)')

% Part 4
P_signals_10 = sum(diag(s_A_10(1:3,1:3)^2));
P_noise_10 = s_A_10(4,4)^2;

SNR_10 = P_signals_10/P_noise_10;
SNR_ideal_10 = sum(beta(:,1))/noise_var;

% Part 5
norm_R_R_hat_10 = norm(R_10-R_hat_10);
part_5_works_10 = norm_R_R_hat_10 < D(4,4);

P_s_10 = V_10(:,1:3)*V_10(:,1:3)';
P_n_10 = eye(length(P_s_10)) - P_s_10;

% Part 6
mvdr_10 = zeros(length(Theta_Phi),1);
music_10 = zeros(length(Theta_Phi),1);

for index = 1:length(Theta_Phi)
    [S_index,~] = pset01a(Theta_Phi(:,index),r(11:31,:),ones(1,N),d_lam,noise_var);
    s_index = S_index/norm(S_index);
    mvdr_10(index) = real(1/(s_index'*pinv(R_hat_10)*s_index));
    music_10(index) = real(1/(s_index'*P_n_10*s_index));
end

MVDR_10 = reshape(mvdr_10,length(theta),length(phi));
MUSIC_10 = reshape(music_10,length(theta),length(phi));

figure(12)
subplot(2,1,1)
contour(theta,phi,MVDR_10)
title('MVDR Spectrum (N=10)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MVDR}$$','interpreter','latex');
colorbar
subplot(2,1,2)
surf(theta,phi,MVDR_10,'EdgeColor','interp')
title('MVDR Spectrum (N=10)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MVDR}$$','interpreter','latex');
colorbar

figure(13)
subplot(2,1,1)
contour(theta,phi,MUSIC_10)
title('MUSIC Spectrum (N=10)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MUSIC}$$','interpreter','latex');
colorbar
subplot(2,1,2)
surf(theta,phi,MUSIC_10,'EdgeColor','interp')
title('MUSIC Spectrum (N=10)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MUSIC}$$','interpreter','latex');
colorbar

% Part 7
min_MVDR_10 = 1/norm(pinv(R_hat_10));
min_MUSIC_10 = 1/norm(P_n_10);

% Part 8
MVDR_source_10 = zeros(size(Theta,2),1);
MUSIC_source_10 = zeros(size(Theta,2),1);

for index = 1:size(Theta,2)
    s_index = S_10(:,index)/norm(S_10(:,index));
    MVDR_source_10(index) = real(1/(s_index'*pinv(R_hat_10)*s_index));
    MUSIC_source_10(index) = real(1/(s_index'*P_n_10*s_index));
end

MVDR_peak_to_min_10 = MVDR_source_10/min_MVDR_10;
MUSIC_peak_to_min_10 = MUSIC_source_10/min_MUSIC_10;

% Part 9
min_grid_MVDR_10 = min(min(MVDR_10));
min_grid_MUSIC_10 = min(min(MUSIC_10));

delta_min_MVDR_10 = min_grid_MVDR_10-min_MVDR_10;
delta_min_MUSIC_10 = min_grid_MUSIC_10-min_MUSIC_10;

% I don't see any major changes...

%% Part 11:

% Set up
N = 100;

theta_shift = deg2rad([10,10,30]); phi_shift = deg2rad([20,10,150]);
Theta_shift = [theta_shift;phi_shift];

[S_shift,A_shift] = pset01a(Theta_shift,r,beta,d_lam,noise_var);

R_shift = S_shift*diag(beta(:,1))*S_shift' + noise_var*eye(length(S_shift));
R_hat_shift = pset01b(A_shift);

% Part 1
[V_shift,D_shift] = eig(R_shift);
sing_vals_shift = sqrt(diag(D_shift));

figure(14)
stem(real(sing_vals_shift))
title('Theoretical Singular Values of A (Shifted Source)')

% Part 3
[u_A_shift,s_A_shift,v_A_shift] = svd(A_shift);

figure(15)
stem(diag(s_A_shift));
title('Singular Values of A (Shifted Source)')

% Part 4
P_signals_shift = sum(diag(s_A_shift(1:3,1:3)^2));
P_noise_shift = s_A_shift(4,4)^2;

SNR_shift = P_signals_shift/P_noise_shift;
SNR_ideal_shift = sum(beta(:,1))/noise_var;

% Part 5
norm_R_R_hat_shift = norm(R_shift-R_hat_shift);
part_5_works_shift = norm_R_R_hat_shift < D(4,4);

P_s_shift = V_shift(:,1:3)*V_shift(:,1:3)';
P_n_shift = eye(length(P_s_shift)) - P_s_shift;

% Part 6
gridsize = 100;
theta_domain = linspace(0,deg2rad(2),gridsize);
phi_domain = linspace(0,deg2rad(2),gridsize);
[theta,phi] = meshgrid(theta_domain,phi_domain);
Theta_Phi = [theta(:).';phi(:).'];

mvdr_shift = zeros(length(Theta_Phi),1);
music_shift = zeros(length(Theta_Phi),1);

for index = 1:length(Theta_Phi)
    [S_index,~] = pset01a(Theta_Phi(:,index),r,ones(1,N),d_lam,noise_var);
    s_index = S_index/norm(S_index);
    mvdr_shift(index) = real(1/(s_index'*pinv(R_hat_shift)*s_index));
    music_shift(index) = real(1/(s_index'*P_n_shift*s_index));
end

MVDR_shift = reshape(mvdr_shift,length(theta),length(phi));
MUSIC_shift = reshape(music_shift,length(theta),length(phi));

figure(16)
subplot(2,1,1)
contour(theta,phi,MVDR_shift)
title('MVDR Spectrum (Shifted Source)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MVDR}$$','interpreter','latex');
colorbar
subplot(2,1,2)
surf(theta,phi,MVDR_shift,'EdgeColor','interp')
title('MVDR Spectrum (Shifted Source)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MVDR}$$','interpreter','latex');
colorbar

figure(17)
subplot(2,1,1)
contour(theta,phi,MUSIC_shift)
title('MUSIC Spectrum (Shifted Source)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MUSIC}$$','interpreter','latex');
colorbar
subplot(2,1,2)
surf(theta,phi,MUSIC_shift,'EdgeColor','interp')
title('MUSIC Spectrum (Shifted Source)')
xlabel('$$\theta$$','interpreter','latex'); 
ylabel('$$\phi$$','interpreter','latex'); 
zlabel('$$S_{MUSIC}$$','interpreter','latex');
colorbar

% Part 7
min_MVDR_shift = 1/norm(pinv(R_hat_shift));
min_MUSIC_shift = 1/norm(P_n_shift);

% Part 8
MVDR_source_shift = zeros(size(Theta,2),1);
MUSIC_source_shift = zeros(size(Theta,2),1);

for index = 1:size(Theta,2)
    s_index = S_shift(:,index)/norm(S_shift(:,index));
    MVDR_source_shift(index) = real(1/(s_index'*pinv(R_hat_shift)*s_index));
    MUSIC_source_shift(index) = real(1/(s_index'*P_n_shift*s_index));
end

MVDR_peak_to_min_shift = MVDR_source_shift/min_MVDR_shift;
MUSIC_peak_to_min_shift = MUSIC_source_shift/min_MUSIC_shift;

% Part 9
min_grid_MVDR_shift = min(min(MVDR_shift));
min_grid_MUSIC_shift = min(min(MUSIC_shift));

delta_min_MVDR_shift = min_grid_MVDR_shift-min_MVDR_shift;
delta_min_MUSIC_shift = min_grid_MUSIC_shift-min_MUSIC_shift;
