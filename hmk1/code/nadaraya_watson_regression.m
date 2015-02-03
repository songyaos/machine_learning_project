%nadaraya_watson_regression

clear all ;close all;
[t,X] =  loadData();
X_n =  normalizeData(X);
t = normalizeData(t);
training_size = 100;
test_size = length(t) - training_size;
x_training = X_n(1:training_size,2) ;%only second feature
t_training = t(1:training_size);
x_test = X_n(training_size+1:end,2);%only second feature
t_test = t(training_size+1:end);

h= [0.01, 0.1, 0.25, 1, 2,3,4];
K=10; N =training_size;
Indices = crossvalind('Kfold', N, K);
validation_error= zeros(length(h),K); %no basis function now
for regu = 1: length(h)
    for i = 1:10
        current_x_training = x_training(Indices ~= i);
        current_t_training = t_training(Indices ~= i);
        current_x_valid = x_training(Indices == i);
        current_t_valid = t_training(Indices == i);
        %h(regu)
        u =  bsxfun(@minus, current_x_training',current_x_valid);
        g_u = tri_kernel(u,h(regu));
        estimated_value =sum(diag(current_t_valid)*g_u,2)./sum(g_u,2);
        %estimated_value(isnan(estimated_value)) = mean(current_t_training);
        validation_error(regu,i) = sqrt(  1/K *   sum(      (current_t_valid - estimated_value).^2   )) ;
        
    end
end
figure;set(gca,'FontSize',20);
plot (h, mean(validation_error,2),'or-');
xlabel('kernel width');
ylabel('average validation set error');
