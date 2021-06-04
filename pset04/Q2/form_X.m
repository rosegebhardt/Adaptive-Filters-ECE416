function [x,X] = form_X(h,s,noise_var,N,M)

v = sqrt(noise_var)*randn(N,1);
x = zeros(N,1);
X = zeros(M,N-M+1);

x(1) = v(1); 
x(2) = v(2) + h(1)*s(1); 
x(3) = v(3) + h(1)*s(2) + h(2)*s(1);

for index = 4:N
    x(index) = v(index) + h(1)*s(index-1) + h(2)*s(index-2) + h(3)*s(index-3);
end

for index = 1:(N-M+1)
    X(:,index) = flipud(x(index:(index+M-1)));
end

end