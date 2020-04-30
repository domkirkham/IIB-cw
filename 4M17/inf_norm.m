function [] = inf_norm(A, b)


% inf form
A_tilde = [-A; A];

ones_A = ones(size(A_tilde,1), 1);

A_tilde = [A_tilde, ones_A];

zeros_C = zeros(size(A_tilde,2)-1, 1);

c = [zeros_C; 1];

b_tilde = [-b; b];

options = optimoptions('linprog','Algorithm','dual-simplex');

[x_inf,fval,exitflag,output] = linprog(c, -A_tilde, -b_tilde);

inf_residuals = A*x_inf(1:size(A,2)) - b;

histogram(inf_residuals)
xlabel("Ax_{i}-b_{i}")
ylabel("Density")