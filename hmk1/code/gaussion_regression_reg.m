%gaussian regression reg
clear all ;close all;
[t,X] =  loadData();
X_n =  normalizeData(X);
t = normalizeData(t);
training_size = 100;
test_size = length(t) - training_size;
x_training = X_n(1:training_size,:) ;
t_training = t(1:training_size);
x_test = X_n(training_size+1:end,:);
t_test = t(training_size+1:end);

pool = 90;%number of basis functions
s=2;
regularizer = [0, 0.01, 0.1, 1, 10, 100, 1000];
K=10; N =training_size;
basis = 'gaussian';
w_ml = zeros(length(regularizer),K,pool+1);
Indices = crossvalind('Kfold', N, K);
training_error = zeros(length(regularizer),K);%the validate set error matrix
for regu = 1: length(regularizer)
    for i = 1:10
        Mu = datasample(x_training(Indices ~= i), pool,  'Replace', false);%sample basis function
        phi =  designMatrix(x_training(Indices ~= i),basis,Mu, s);%construct matrix
        w_ml(regu,i,:) =  pinv(  regularizer(regu)*eye(size(phi,2))  +  (phi') * phi    ) * phi' *  t_training(Indices ~= i);%obtain coef
        %training error
        regression_value1 = designMatrix(x_training(Indices == i), basis, Mu,s) * reshape( w_ml(regu,i,:), length(w_ml(regu,i,:)),1);
        training_error(regu,i) =sqrt(  1/training_size *   sum(      (t_training(Indices == i) - regression_value1).^2   )) ;
        
    end
end
figure;set(gca,'FontSize',20);
semilogx(regularizer, mean(training_error,2),'ro-');
xlabel('regularizer');
ylabel('average validation set error');
