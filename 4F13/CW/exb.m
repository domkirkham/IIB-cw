function [plotcoords] = exb(x, y)

meanfunc = [];
covfunc = @covSEiso;
likfunc = @likGauss;



%x = gpml_randn(0.8, 20, 1);                 % 20 training inputs

%y = sin(3*x) + 0.1*gpml_randn(0.9, 20, 1);  % 20 noisy training targets
%xs = linspace(-3, 3, 61)';   

training_liks = [];
params = [];

for length = -4:1:4
    for sf = -4:1:4
        for liknoise = -4:1:4

            hyp = struct('mean', [], 'cov', [length sf], 'lik', liknoise);

            hyp2 = minimize(hyp, @gp, -75, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
            
            [nlZ dnlZ] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
            
            training_liks = [training_liks, nlZ];
            
            params = [params, [length, sf, liknoise]'];
        end
    end
end

plotcoords = [params; training_liks];
            
xs = linspace(min(x)-1.5, max(x)+1.5, 1000)';
   
[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
            % To plot the predictive mean at the test points together with the predictive 95% confidence bounds and the training data

f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
figure(1);
axes('Box','off', 'Units','inches','Position',[1.5 1.5 8 6]);
hold on; 
fill([xs; flip(xs,1)], f, [8 7 7]/8);
plot(xs, mu, 'r--'); 

plot(x, y, 'blacko');
%ycovmin = exp(-0.1087)*exp(-xs.*xs/ (2 * exp(-2.0540))) ;
%ycov = exp(-xs.*xs/ (2 * exp(-1))) ;
%plot(xs, ycovmin, 'r-');
%plot(xs, ycov, 'b-');
xlim([-4.2, 3.8]);
xlabel('x');
ylabel('y');
legend('95% confidence bounds', 'Predictive mean at test points', 'Training points');
hold off;

figure(2);
scatter3(-2.0540, -0.1087, -2.1385, 75, 'red', 'filled');
hold on;
scatter3(2.0845, -0.3625, -0.4109, 75, 'red', 'filled');
scatter3(plotcoords(1,:), plotcoords(2,:), plotcoords(3,:), 50, plotcoords(4,:), 'filled');
cb = colorbar;
cb.Label.String = "negative log marginal likelihood";
colormap(copper);
xlabel("log(l)");
ylabel("log(\sigma_{f})");
zlabel("log(\sigma_{noise})");