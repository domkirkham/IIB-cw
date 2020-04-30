function [] = one_norm(A, b)
% 1-norm

% Form identity matrices to append to A
ones_A = eye(size(A,1));

% Make A_tilde matrix
A_tilde = [-A, ones_A; A, ones_A];

% Make c
zeros_C = zeros(size(A,2), 1);
ones_C = ones(size(A,1), 1);

c = [zeros_C; ones_C];

% Make b
b_tilde = [-b; b];

% Set method to dual simplex
options = optimoptions('linprog','Algorithm','dual-simplex');

% Solve the linprog problem
[x_inf,fval,exitflag,output] = linprog(c, -A_tilde, -b_tilde);

%Calculate and plot residuals
inf_residuals = A*x_inf(1:size(A,2)) - b;

histogram(inf_residuals)
xlabel("Ax_{i}-b_{i}")
ylabel("Density")