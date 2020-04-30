function [] = backtrack_1norm(A, b)

% Form all the required matrices and vectors as in the original l1 problem
ones_A = eye(size(A,1));

A_tilde = [-A, ones_A; A, ones_A];

resids = [];
min_errors = [];

zeros_C = zeros(size(A,2), 1);
ones_C = ones(size(A,1), 1);

c = [zeros_C; ones_C];

b_tilde = [-b; b];

% Initialisation of x at a point far inside the feasible set
x = [zeros(size(A,2), 1); 100*ones(size(A,1), 1)];

% Set parameters for backtracking line search
alpha = 0.5;
beta = 0.8;
e = 1;
k = 1;
num_its = 0;

% Basic algorithm with some attempts to avoid the large gradient problem
% which occurs close to the optimum
while size(A,1)/k > 200
    while sqrt(e) > 0.005 && num_its < 100
            t=1;
            grad = c;

            % Loop over each constraint
            for i = 1:size(A_tilde,1)
                % Threshold residuals to avoid large gradients
                denom = (A_tilde(i,:)*x - b_tilde(i));
                if abs(denom) < 0.1
                    denom = sign(denom)*0.1;
                end
                grad = grad - (A_tilde(i,:)' ./ denom) / k;
            end

            dx = -grad;
            
            % Magnitude of gradient
            % Not actually useful since gradients become large close to
            % optimum
            e = grad'*grad;

            % backtracking
            while (c'*x + alpha*t*grad'*dx) <= (c'*(x + t*dx))
                t = t * beta;
            end

            % update x
            x = x + t*dx;

            num_its = num_its + 1;

            % collect values for plots
            resid = abs(A*x(1:size(A,2)) - b);
            new_resid = sum(resid);
            min_error = f_l1(x, A_tilde, b_tilde, c);

            resids = [resids, new_resid];
            min_errors = [min_errors, min_error];
    end
    k = k*10;
    num_its = 0;
end
min(min_errors)
min_errors = min_errors - min(min_errors);

%inf_residuals = A*x(1:size(A,2)) - b;
%histogram(inf_residuals)

semilogy(min_errors)
xlabel("Step")
ylabel("f(x^{(k)}) - f(x^{*})")
