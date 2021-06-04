function [w_RLS,P_RLS,xi] = RLS(lam,delta,x,u)

[M,N] = size(u);
w_hat = zeros(M,1); 
w_hat(ceil(M/2)) = 1;
xi = zeros(1,N);

P = (1/delta)*eye(M);

for index = 1:N
    k = (1/lam)*P*u(:,index)/(1 + (1/lam)*u(:,index)'*P*u(:,index));
    d = x(index+ceil(M/2)-1);
    xi(index) = d - w_hat'*u(:,index);
    w_hat = w_hat + k*xi(index)';
    P = (1/lam)*P - (1/lam)*k*u(:,index)'*P;
end

w_RLS = w_hat;
P_RLS = P;

end