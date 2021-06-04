function [w_LMS,e,eps] = LMS(mu,w0,beta,u,N,type)

[M,~] = size(u);
w_hat = zeros(M,1);
e = zeros(1,N);
eps = zeros(M,N);

for index = 1:N
    y = w_hat'*u(:,index);
    d = beta.'*u(:,index) + randn;
    e(index) = d-y;
    eps(:,index) = w0 - w_hat;
    if strcmp(type,'LMS')
        % LMS Update
        w_hat = w_hat + mu*u(:,index)*e(index)';
    elseif strcmp(type,'NLMS')
        % NLMS Update
        w_hat = w_hat + (mu*u(:,index)*e(index)')/(u(:,index)'*u(:,index));
    else
        warning('Choose type LMS or NLMS')
    end       
end

w_LMS = w_hat;

end