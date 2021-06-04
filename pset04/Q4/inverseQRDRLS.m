function [w_hat,e] = inverseQRDRLS(lam,delta,w_q,A)

[M,N] = size(A);
w_hat = zeros(M,N);
e = zeros(N,1);

root_P = (1/sqrt(delta))*eye(M);
W_hat = zeros(M,1);

for index = 1:N
    
    prearray = [1,(1/(sqrt(lam)))*A(:,index)'*root_P;zeros(M,1),(1/(sqrt(lam)))*root_P];
    [~,r] = qr(prearray'); postarray = r';
    
    k = postarray(2:M+1,1)*(1/postarray(1,1));
    e(index) = w_q'*A(:,index) - W_hat'*A(:,index);
    W_hat = W_hat + k*e(index)';
    w_hat(:,index) = W_hat;
    
end

end

