function [w,e] = GSC(mu,S,A,g)

[M,L] = size(S);
[~,N] = size(A);

w_q = S*pinv(S'*S)*g;
w = zeros(M,N,L);
e = zeros(N,L);

for index = 1:L
    [w(:,:,index),e(:,index)] = LMS(mu,w_q(:,index),A,'LMS');
end

end