function [] = backtrack_newton(A, b)

n = size(A,2);
x = zeros(n, 1);
u = ones(n, 1);
f = [x-u;-x-u];

ALPHA = 0.01;
BETA = 0.5;
num_its = 0;
lambda = 0.01 * max(2*A'*b);
t0 = min(max(1,1/lambda),2*n/1e-3);
s = Inf;
pobj  = Inf; 
dobj  =-Inf;
MAX_LS_ITER = 200;
resids = [];

t=t0;

while num_its < 200
    
    z = A*x - b;
    
    nu = 2*z;

    maxAnu = norm(A'*nu,inf);
    if (maxAnu > lambda)
        nu = nu*lambda/maxAnu;
    end
    pobj  =  z'*z+lambda*norm(x,1);
    dobj  =  max(-0.25*nu'*nu-nu'*b,dobj);
    gap   =  pobj - dobj;
    
    if (s >= 0.5)
        t = max(min(2*n*2/gap, 2*t), t);
    end
    


    q1 = 1./(u+x);          q2 = 1./(u-x);
    d1 = diag((q1.^2+q2.^2)/t);   d2 = diag((q1.^2-q2.^2)/t);
   
    % calculate gradient
    gradphi = [A'*(z*2)-(q1-q2)/t; lambda*ones(n,1)-(q1+q2)/t];
    
    % calculate Hessian
    hessphi = [2*t*A'*A + d1, d2; d2, d1];
    
    % calculate step size
    L = chol(hessphi);
    dxu = -inv(L)'*inv(L)*gradphi;
    dx = dxu(1:n);
    du = dxu(n+1:end);
    
    phi = z'*z + lambda*sum(u) - sum(log(-f))/t;
    s = 1.0;
    gdx = gradphi'*dxu;
    for lsiter = 1:MAX_LS_ITER
        newx = x+s*dx; newu = u+s*du;
        newf = [newx-newu;-newx-newu];
        if (max(newf) < 0)
            newz   =  A*newx-b;
            newphi =  newz'*newz+lambda*sum(newu)-sum(log(-newf))/t;
            if (newphi-phi <= ALPHA*s*gdx)
                break;
            end
        end
        s = BETA*s;
    end
    if (lsiter == MAX_LS_ITER) break; end % exit by BLS
        
    x = newx; u = newu; f = newf;
    
    num_its = num_its + 1;

    resid = abs(A*x - b);
    new_resid = sum(resid);
    %min_error = f_l1(x, A_tilde, b_tilde, c);

    resids = [resids, new_resid];
    %min_errors = [min_errors, min_error];
end
%min(min_errors)
%min_errors = min_errors - min(min_errors);

%inf_residuals = A*x(1:size(A,2)) - b;
plot(resids)

%semilogy(min_errors)