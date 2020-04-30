function [f] = f_l1(x, A, b, c)

f = c'*x;

for i = 1:size(A,1)
    f = f - log(A(i,:)*x - b(i));
end