function[k,p,a] = pset02_LD(r,N)

% Initialize values to be returned
k = zeros(N,1);
p = zeros(N,1);
a = zeros(N,N);

% Initialize intermediate values
aB = zeros(N,N);
Delta = zeros(N,1);

% Initial values
a(1,1) = 1;
aB(1,1) = a(1,1)';
p(1) = r(1);

% Assumes r is a row vector --> useful for defining Delta
flip_r = conj(fliplr(r));
% size(flip_r)

% Levinson-Durbin Recursion
for index = 1:N
    Delta(index) = flip_r(end-index:end-1)*a(1:index,index);
    k(index+1) = -Delta(index)/p(index);
    p(index+1) = p(index)*(1-abs(k(index+1))^2);
    a(1:(index+1),index+1) = [a(1:index,index);0] + k(index+1)*[0;aB(1:index,index)];
    aB(1:(index+1),index+1) = flipud(a(1:(index+1),index+1))';
    % disp("Index: "+index)
    % disp(a)
    % disp(aB)
end
    


