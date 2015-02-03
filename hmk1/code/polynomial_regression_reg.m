%polynomial regression reg
%the best regularizer depends on the way to divide the 10 fold
clear all;close all;
[t,X] =  loadData();
X_n =  normalizeData(X);
t = normalizeData(t);
training_size = 100;
test_size = length(t) - training_size;
x_training = X_n(1:training_size,2) ;%only second feature
t_training = t(1:training_size);
x_test = X_n(training_size+1:end,2);%only second feature
t_test = t(training_size+1:end);

degree = 8;
regularizer = [0, 0.01, 0.1, 1, 10, 100, 1000];
K=10; N =training_size;
basis = 'polynomial';

Indices = crossvalind('Kfold', N, K);
w_ml = zeros(length(regularizer),K,degree*size(x_training,2) +1);%do not use length if not a vector, dangerous
train_error = zeros(length(regularizer),K);%the validate set error matrix
for regu = 1: length(regularizer)
    for i = 1:10
        phi =  designMatrix(x_training(Indices ~= i), basis, degree);
        w_ml(regu,i,:) = pinv(    regularizer(regu)*eye(size(phi,2))  +  (phi') * phi    ) * phi' *  t_training(Indices ~= i);
        regression_value = designMatrix(x_training(Indices == i), basis, degree) * reshape( w_ml(regu,i,:), length(w_ml(regu,i,:)),1);
        train_error(regu,i) =  sqrt(  1/K *   sum(      (t_training(Indices == i) - regression_value).^2   )                 );
        %sqrt(1/training_size * sum( (y_d1 - t_training).^2  )  );
    end
end
figure;set(gca,'FontSize',20);
%plot(regularizer(1), mean(train_error(1,:)),'ro-')
semilogx(regularizer, mean(train_error,2),'ro-');

xlabel('regularizer');
ylabel('average validation set error');



