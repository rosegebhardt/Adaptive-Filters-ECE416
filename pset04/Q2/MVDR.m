function [w,e] = MVDR(lam,delta,S,A)

[M,L] = size(S);
[~,N] = size(A);

w_q = S*pinv(S'*S);
w = zeros(M,N,L);
e = zeros(N,L);

for index = 1:L
    [w(:,:,index),e(:,index)] = RLS_beam(lam,delta,w_q(:,index),A);
end

end
