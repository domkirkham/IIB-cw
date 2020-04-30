function [] = min_energy(A, b)


% 2 norm
fun = @(x) x'*x;
x0 = zeros(size(A,2), 1);
x = fmincon(fun, x0, [], [], A'*A, A'*b);

plot(x)
xlabel("i")
ylabel("x_{i}")