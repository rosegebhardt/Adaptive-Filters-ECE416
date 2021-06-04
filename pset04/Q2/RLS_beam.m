function [w_RLS,xi] = RLS(lam,delta,w_q,u)

[M,N] = size(u);
w_hat = zeros(M,N);
xi = zeros(N,1);

P = (1/delta)*eye(M);

for index = 1:N-1
    k = (1/lam)*P*u(:,index)/(1 + (1/lam)*u(:,index)'*P*u(:,index));
    y = w_hat(:,index)'*u(:,index);
    d = w_q'*u(:,index);
    xi(index) = d-y;
    w_hat(:,index+1) = w_hat(:,index) + k*xi(index)';
    P = (1/lam)*P - (1/lam)*k*u(:,index)'*P;    
end

w_RLS = w_hat;

end