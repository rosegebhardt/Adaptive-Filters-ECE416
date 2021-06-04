function [w_hat,e] = QRDRLS(lam,delta,w_q,A)

[M,N] = size(A);
w_hat = zeros(M,N);
e = zeros(N,1);

root_phi = sqrt(delta)*eye(M);
p = zeros(M,1);

for index = 1:N
    
    prearray = [sqrt(lam)*root_phi,A(:,index); sqrt(lam)*p',w_q'*A(:,index);zeros(1,M),1];
    [~,r] = qr(prearray'); postarray = r';
    
    root_phi = postarray(1:M,1:M);
    p = postarray(M+1,1:M)';
    
    w_hat(:,index) = (p'*pinv(root_phi))';
    e(index) = w_q'*A(:,index) - w_hat(:,index)'*A(:,index);
    
end

end

