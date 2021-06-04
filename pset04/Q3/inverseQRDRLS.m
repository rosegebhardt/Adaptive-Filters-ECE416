function [w_hat,e] = inverseQRDRLS(lam,delta,A,d,n)

[M,N] = size(A);
if n > N
    error("N is greater than the number of samples!");
end

root_P = (1/sqrt(delta))*eye(M);
w_hat = zeros(M,1);

for index = 1:n
    prearray = [1,(1/(sqrt(lam)))*A(:,index)'*root_P;zeros(M,1),(1/(sqrt(lam)))*root_P];
    [~,r] = qr(prearray'); postarray = r';
    
    k = postarray(2:M+1,1)*(1/postarray(1,1));
    e = d(index+ceil(M/2)-1) - w_hat'*A(:,index);
    w_hat = w_hat + k*e';
end

end

