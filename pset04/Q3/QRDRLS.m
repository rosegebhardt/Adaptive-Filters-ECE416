function [w_hat,e] = QRDRLS(lam,delta,A,d,n)

[M,N] = size(A);
if n > N
    error("N is greater than the number of samples!");
end

root_phi = sqrt(delta)*eye(M);
p = zeros(M,1);

for index = 1:n
    prearray = [sqrt(lam)*root_phi,A(:,index); sqrt(lam)*p',d(index+ceil(M/2)-1);zeros(1,M),1];
    [~,r] = qr(prearray'); postarray = r';
    
    root_phi = postarray(1:M,1:M);
    p = postarray(M+1,1:M)';
end

w_hat = (p'*pinv(root_phi))';
e = d(n+ceil(M/2)-1) - w_hat'*A(:,n);

end

