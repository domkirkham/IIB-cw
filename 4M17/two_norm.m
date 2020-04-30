function [] = two_norm(A, b)


% 2 norm

x = lsqminnorm(A, b);

inf_residuals = A*x - b;

sqrt(sum(inf_residuals.^2))

histogram(inf_residuals)
xlabel("Ax_{i}-b_{i}")
ylabel("Density")