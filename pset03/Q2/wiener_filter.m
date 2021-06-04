function w_wiener = wiener_filter(s,x,M)

r = zeros(M,1);
p = zeros(M,1);

for index = 1:M
    r(index) = mean(conj(x(1:end-index+1)).*x(index:end));
    p(index) = mean((s(1+10:end-index+1))'.*x(index:end-10));
end

R = toeplitz(r);
w_wiener = flip(R\p);