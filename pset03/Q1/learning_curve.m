function [J,D] = learning_curve(mu,w0,beta,u,N,num_runs,type)

J_sum = zeros(1,N);
D_sum = zeros(1,N);

for index = 1:num_runs    
    [~,e,eps] = LMS(mu,w0,beta,u,N,type);
    J_sum = J_sum + e.^2;
    D_sum = D_sum + vecnorm(eps).^2;
end

J = J_sum/num_runs;
D = D_sum/num_runs;