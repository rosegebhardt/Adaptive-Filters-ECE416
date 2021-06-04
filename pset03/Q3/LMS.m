function [w_LMS,e] = LMS(mu,w_q,u,type)

[M,N] = size(u);
w_hat = zeros(M,N);
e = zeros(N,1);

for index = 1:N-1
    y = w_hat(:,index)'*u(:,index);
    d = w_q'*u(:,index);
    e(index) = d-y;
    if strcmp(type,'LMS')
        % LMS Update
        w_hat(:,index+1) = w_hat(:,index) + mu*u(:,index)*e(index)';
    elseif strcmp(type,'NLMS')
        % NLMS Update
        w_hat(:,index+1) = w_hat(:,index) + (mu*u(:,index)*e(index)')/(u(:,index)'*u(:,index));
    else
        warning('Choose type LMS or NLMS')
    end       
end

w_LMS = w_hat;

end