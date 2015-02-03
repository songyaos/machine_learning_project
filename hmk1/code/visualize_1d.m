function [] = visualize_1d(X_n, X_train, X_test, t_train, t_test, degree)
% X_n is 1-d
% X_train, X_test, t_train, t_test should all be 1-d, and need to be defined as well.
% You should modify y_ev


% Plot a curve showing learned function.
x_ev = (min(X_n):0.1:max(X_n))';

% Put your regression estimate here.
phi=  designMatrix(X_train, 'polynomial', degree);%training using train data
w_ml = pinv( (phi') * phi) * phi' * t_train;%get the train coef
phi_xev = designMatrix(x_ev, 'polynomial', degree);%estimate the output given input x_ev.
y_ev = phi_xev * w_ml ;%regression result 

%size(x_ev), size(y_ev)
figure;
% Make the fonts larger, good for reports.
set(gca,'FontSize',15);
plot(x_ev,y_ev,'r.-');  
hold on;
plot(X_train,t_train,'g.');
plot(X_test,t_test,'bo');
hold off;
title(sprintf('Fit with degree %d polynomial',degree));
