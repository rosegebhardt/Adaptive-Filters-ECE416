function [x_est,x_pred,K_est,K_pred,G] = squarerootKalman(x_prev,K_prev,y,A,C,Qv,Qw)

% x_prev = x(n|n-1), K_prev = K(n,n-1)
% x_est = x(n|n), K_est = K(n,n)
% x_pred = x(n+1|n), K_pred = K(n+1|n)

[N,~] = size(x_prev); [p,~] = size(y);

S = chol(K_prev,'lower');
sqrt_Qv = chol(Qv,'lower'); sqrt_Qw = chol(Qw,'lower');

E = [sqrt_Qw,C*S,zeros(p,N);zeros(N,p),A*S,sqrt_Qv];
[~,r] = qr(E'); F = r';

T = F(1:p,1:p)*F(1:p,1:p)';
alpha = y - C*x_prev;
G = K_prev*C'*pinv(T);

K_est = (eye(4)-G*C)*K_prev;
K_pred = F(p+1:p+N,p+1:p+N)*F(p+1:p+N,p+1:p+N)';

if (min(eig(K_est)) < 0)
    warning("WARNING: Condition K(n,n) > 0 failed.")
end

x_est = x_prev + G*alpha;
x_pred = A*x_est;

end