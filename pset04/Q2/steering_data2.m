function [S,A] = steering_data2(Theta,r,beta,d_lam,noise_var)

% COMMENTS:
% Let Theta be a 2 by L matrix: 
    % row 1 is theta, row 2 is phi
    % columns correspond to sources
% Let r be an M by 3 matrix
    % rows correspond to x, y, and z components
    % columns corresond to each sensor in the array
% Let beta be weights for each source at each time, should be L by N
    % rows are over increasing time
    % columns are each element in the array at one time
% Let d_lam = d/lambda be a constant (narrowband)
% Add dimension checks/warnings?

% Find # of sources (L), array elements (M), time steps (N)
[~,L] = size(Theta);
[M,~] = size(r);
[~,N] = size(beta);

% Initialize arrays
a = zeros(3,L);
k = zeros(3,L);
S = zeros(M,L);

% Fill in arrays, generate S matrix
for index = 1:L
    % Unit directional vector
    a(:,index) = [0;0;cos(Theta(1,index))];
    % Wavenumber vector
    k(:,index) = (2*pi*d_lam)*a(:,index);
    % Steering vector
    S(:,index) = (1/sqrt(M))*exp(-1*1j*r*k(:,index));
end

% Matrix of unit variance complex Gaussian variables
v = (sqrt(1/2))*sqrt(noise_var)*(randn(M,N)+1i*randn(M,N));

% Add noise to beta
beta = beta + (sqrt(1/2))*(randn(L,N)+1i*randn(L,N));

% Data matrix
A = S*beta + v;