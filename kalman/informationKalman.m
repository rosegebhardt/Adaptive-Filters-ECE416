function [chi_est,chi_pred,P_est,P_pred] = informationKalman(chi_prev,P_prev,y,A,C,Qv,Qw)

% chi_prev = chi(n,n-1), P_prev = P(n,n-1)
% chi_est = x(n,n), P_est = P(n,n)
% chi_pred = x(n+1,n), P_pred = P(n+1|n)

chi_est = chi_prev + C'*pinv(Qw)*y;
P_est = P_prev + C'*pinv(Qw)*C;

M = pinv(A')*P_est*pinv(A);
F = pinv(eye(4)+M*Qv);

P_pred = F*M;
chi_pred = F*pinv(A')*chi_est;

end

