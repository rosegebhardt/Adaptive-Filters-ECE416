function R = pset01b(A)
[~,N] = size(A);
% Note - If N = 1, this is the correlation matrix of u[n]
R = (1/N)*(A*A');