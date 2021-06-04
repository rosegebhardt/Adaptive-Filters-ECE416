function [w,e] = MVDR(lam,delta,S,A,type)

[M,L] = size(S);
[~,N] = size(A);

w_q = S*pinv(S'*S);
w = zeros(M,N,L);
e = zeros(N,L);

if strcmp(type,'QRDRLS')
    
    for index = 1:L
        [w(:,:,index),e(:,index)] = QRDRLS(lam,delta,w_q(:,index),A);
    end
    
elseif strcmp(type,'Inverse QRDRLS')
    
    for index = 1:L
        [w(:,:,index),e(:,index)] = inverseQRDRLS(lam,delta,w_q(:,index),A);
    end
    
else
    warning('Specify QRDRLS or Inverse QRDRLS');
end

end
