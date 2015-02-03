%1d visulization
clear all;
[t,X] =  loadData();
X_n_d =  normalizeData(X);%The orginal data matrix
t = normalizeData(t);
X_n = X_n_d(:,2);%truancated matrix,only consider the 2nd feature
training_size = 100;
test_size = length(t) - training_size;
X_train = X_n_d(1:training_size,2) ;
t_train = t(1:training_size);
X_test = X_n_d(training_size+1:end,2);
t_test = t(training_size+1:end);


%degree = 5;
for degree  = 1:7
    visualize_1d(X_n, X_train, X_test, t_train, t_test, degree);
end