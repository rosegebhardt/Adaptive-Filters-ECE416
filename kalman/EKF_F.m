function [f,F] = EKF_F(x)

f = [10*pi*sin(x(1)*x(3));10*pi*sin(x(2)*x(4));10*pi*sin(x(3));10*pi*sin(x(4))];

F = [10*pi*x(3)*cos(x(1)*x(3)),0,10*pi*x(1)*cos(x(1)*x(3)),0;
     0,10*pi*x(4)*cos(x(2)*x(4)),0,10*pi*x(2)*cos(x(2)*x(4));
     0,0,10*pi*cos(x(3)),0;
     0,0,0,10*pi*cos(x(4))];

end