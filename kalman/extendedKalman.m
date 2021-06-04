function [x_est,x_pred,K_est,K_pred] = extendedKalman(x_prev,K_prev,y,Qv,Qw,fCase,hCase)

% x_prev = x(n|n-1), K_prev = K(n,n-1)
% x_est = x(n|n), K_est = K(n,n)
% x_pred = x(n+1|n), K_pred = K(n+1|n)

if (hCase == 1)
    h = sin(x_prev); H = diag(cos(x_prev));
elseif (hCase == 2)
    h = atan(x_prev); H = diag(1./(x_prev.^2 + 1));
else
    warning("Enter which case is being used! (1 -> h = sin(x), 2 -> h = atan(x))");
end

alpha = y - h;
S = H*K_prev*H' + Qw;
G = K_prev*H'*pinv(S);

x_est = x_prev + G*alpha;
K_est = (eye(4)-G*H)*K_prev;

if (min(eig(K_est)) < 0)
    warning("WARNING: Condition K(n,n) > 0 failed.")
end

if (fCase == 1)
    [f,F] = EKF_F(x_est);
elseif (fCase == 2)
    [f,F] = EKF_G(x_est);
else
    warning("Enter which case is being used!")
end

x_pred  = f;
K_pred = F*K_est*F' + Qv;

end