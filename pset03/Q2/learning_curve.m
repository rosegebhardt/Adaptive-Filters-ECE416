function [w_LMS,J] = learning_curve(mu,s,x,type,num_runs)

[M,N] = size(x);
w_sum = zeros(M,1);
J_sum = zeros(1,N);

for index = 1:num_runs
    [w,e] = LMS(mu,s,x,type);
    w_sum = w_sum + w;
    J_sum = J_sum + e.^2;
end

w_LMS = w_sum/num_runs;
J = J_sum/num_runs;