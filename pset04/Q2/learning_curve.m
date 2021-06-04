function [w_RLS,J] = learning_curve(lam,delta,x,u,num_runs)

[M,N] = size(u);
w_sum = zeros(M,1);
J_sum = zeros(1,N);

for index = 1:num_runs
    [w,~,e] = RLS_adaptEQ(lam,delta,x,u);
    w_sum = w_sum + w;
    J_sum = J_sum + e.^2;
end

w_RLS = w_sum/num_runs;
J = J_sum/num_runs;