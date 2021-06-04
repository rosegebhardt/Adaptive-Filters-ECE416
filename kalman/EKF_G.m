function [g,G] = EKF_G(x)

A1 = [-0.9,1,0,0;0,-0.9,0,0;0,0,-0.5,0.5;1,0,-0.5,-0.5];
g = abs(A1*x);

eps = 0.1; gd = zeros(4,1);
gd = gd + (x>eps) - (x<-eps);

G = diag(gd);

end