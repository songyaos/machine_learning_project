%gaussian regression
clear all;close all;
[t,X] =  loadData();
X_n =  normalizeData(X);
t = normalizeData(t);

training_size = 100;
test_size = length(t) - training_size;
x_training = X_n(1:training_size,:) ;
t_training = t(1:training_size);
x_test = X_n(training_size+1:end,:);
t_test = t(training_size+1:end);

basis = 'gaussian';
s =2;
pool = 5:10:95;
training_error  = zeros(1,length(pool));
test_error  = zeros(1,length(pool));
for i = 1:length(pool)
    Mu = datasample(x_training, pool(i),  'Replace', false);%sample basis function
    phi =  designMatrix(x_training,basis,Mu, s);%construct matrix
    w_ml =  pinv(  (phi') * phi    ) * phi' *  t_training;%obtain coef
    %training error
    regression_value1 = phi* w_ml;
    training_error(i) =sqrt(  1/training_size *   sum(      (t_training - regression_value1).^2   )) ;
    %test error
    regression_value2 = designMatrix(x_test,basis,Mu, s) *w_ml;
    test_error(i) = sqrt(  1/training_size *   sum(      (t_test - regression_value2).^2   ) );
end
figure;hold on;
set(gca,'FontSize',20);
plot(pool, training_error,'bo-');
plot(pool, test_error,'r*-');
hold off;
xlabel('number of basis functions');
ylabel('errors');
h_legend = legend('training error',  'test error', 0 );
