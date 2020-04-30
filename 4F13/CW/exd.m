function [] = exd()

meanfunc = [];
covfunc = {@covProd, {@covPeriodic, @covSEiso}};
likfunc = @likGauss;



x = linspace(-5,5,200)';            
  
hyp = struct('mean', [], 'cov', [-0.5 0 0 2 0], 'lik', -10);
           
ys = [];
   
K = feval(covfunc{:}, hyp.cov, x);

%mu = feval(meanfunc{:}, hyp.mean, x);
for i = 1:3
    y = chol(K+ 1e-6*eye(200))'*gpml_randn(i, 200, 1);
    ys = [ys, y];
end
            % To plot the predictive mean at the test points together with the predictive 95% confidence bounds and the training data

%f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
figure(1);
axes('Box','off', 'Units','inches','Position',[1.5 1.5 8 6]);
hold on; 
%fill([xs; flip(xs,1)], f, [8 7 7]/8);
%plot(xs, mu, 'r--'); 
plot(x, ys(:,1), 'r-');
xlim([-5.5, 5.5]);
xlabel('x');
ylabel('y');

figure(2);
axes('Box','off', 'Units','inches','Position',[1.5 1.5 8 6]);
hold on; 
%fill([xs; flip(xs,1)], f, [8 7 7]/8);
%plot(xs, mu, 'r--'); 
plot(x, ys(:,2), 'r-');
xlim([-5.5, 5.5]);
xlabel('x');
ylabel('y');

figure(3);
axes('Box','off', 'Units','inches','Position',[1.5 1.5 8 6]);
hold on; 
%fill([xs; flip(xs,1)], f, [8 7 7]/8);
%plot(xs, mu, 'r--'); 
plot(x, ys(:,3), 'r-');
xlim([-5.5, 5.5]);
xlabel('x');
ylabel('y');

figure(4);
imshow(K);