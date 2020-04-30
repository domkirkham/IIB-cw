function [plotcoords] = exe(x, y)

meanfunc = [];
covfunc = {@covSum, {@covSEard, @covSEard}};
likfunc = @likGauss;

hyp = struct('mean', [], 'cov', 0.1*randn(6,1), 'lik', 0);

hyp2 = minimize(hyp, @gp, -75, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

hyp2.cov
hyp2.lik

[nlZ dnlZ] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
            
[x1, x2] = meshgrid(-4:0.1:4,-4:0.1:4);

xs = [x1(:), x2(:)];
   
[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
            % To plot the predictive mean at the test points together with the predictive 95% confidence bounds and the training data
%f = [mu+2*sqrt(s2), flip(mu-2*sqrt(s2),1)];

figure(2);
axes('Box','on', 'Units','inches','Position',[1.5 1.5 10 8.5]);
%fill3([xs(:,1), flip(xs(:,1),1)],[xs(:,2), flip(xs(:,2),1)], f, [8 7 7]/8);
surf(reshape(xs(:,1),81,81),reshape(xs(:,2),81,81),reshape(mu,81,81), 'FaceColor', 'red', 'EdgeColor', 'red'); 
hold on;
s1 = surf(reshape(xs(:,1),81,81),reshape(xs(:,2),81,81),reshape(mu+2*sqrt(s2),81,81), 'FaceColor','black'); 
s2 = surf(reshape(xs(:,1),81,81),reshape(xs(:,2),81,81),reshape(mu-2*sqrt(s2),81,81), 'FaceColor','black');
%surf(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11))
%hold on;
s1.FaceAlpha = 0.15;
s2.FaceAlpha = 0.15;
s1.AlphaData = 0;
s2.AlphaData = 0;
scatter3(x(:,1),x(:,2),y, 'redo');
%ycovmin = exp(-0.1087)*exp(-xs.*xs/ (2 * exp(-2.0540))) ;
%ycov = exp(-xs.*xs/ (2 * exp(-1))) ;
%plot(xs, ycovmin, 'r-');
%plot(xs, ycov, 'b-');

xlabel("x1");
ylabel("x2");
zlabel("y");
view(0,0);
%legend('Predictive mean at test points', '95% confidence bounds');
ylim([-0.00001,0]);
zlim([-2,4]);
%cb = colorbar();
%cb.Label.String = "Predictive variance";
hold off;