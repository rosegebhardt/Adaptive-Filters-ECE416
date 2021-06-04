function U = form_U(a,M,N)

U = zeros(M,N);
% EDIT FF:  your v is real unit variance Gaussian
V = 1/sqrt(2)*(randn(M,N)+1i*randn(M,N));

U(:,1) = V(:,1);
U(:,2) = V(:,2) - a(1)*U(:,1);

for index = 3:N
    U(:,index) = V(:,index) - a(1)*U(:,index-1) - a(2)*U(:,index-2);
end

end


