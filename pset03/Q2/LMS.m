function [w_LMS,e] = LMS(mu,s,x,type)

[M,N] = size(x);
w_hat = zeros(M,1); w_hat(ceil(M/2)) = 1;
e = zeros(1,N);

for index = 1:N
    y = w_hat'*x(:,index);
    d = s(index+ceil(M/2)-1);
    e(index) = d-y;
    if strcmp(type,'LMS')
        % LMS Update
        w_hat = w_hat + mu*x(:,index)*e(index)';
    elseif strcmp(type,'NLMS')
        % NLMS Update
        w_hat = w_hat + (mu*x(:,index)*e(index)')/(x(:,index)'*x(:,index));
    else
        warning('Choose type LMS or NLMS')
    end       
end

w_LMS = w_hat;

end