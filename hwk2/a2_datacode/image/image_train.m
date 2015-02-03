% %hwk2 Q5
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% 	4 -- precomputed kernel (kernel values in training_set_file)
%use cross validation, change kernels ,change parameters
 close all;clear all;
 load test.mat;
 load train.mat;
 c_vector = 150;
% degree_vector = 6:10;
% score_polynomial = zeros(length(c_vector),length(degree_vector));
% for i =1:length(c_vector)
%     for j = 1:length(degree_vector)
%         c=c_vector(i);
%         degree = degree_vector(j);
%         option = ['-s 0 -t 1 -c ' num2str(c) ' -r 1 -d ' num2str(degree) ' -v 10 -q'];
%         fprintf(option);
%         score_polynomial(i,j) = svmtrain(Ltrain, Ftrain,option);
%     end
% end
% save('polynomial.mat','score_polynomial','c_vector','degree_vector');

% gamma_vector = -5:3;
% score_rbf = zeros(length(c_vector),length(gamma_vector));
% for i =1:length(c_vector)
%     for j = 1:length(gamma_vector)
%         c=c_vector(i);
%         gamma = gamma_vector(j);
%         option = ['-s 0 -t 2 -c ' num2str(c) ' -g ' num2str(gamma) ' -v 10 -q'];
%         fprintf(option);
%         score_rbf(i,j) = svmtrain(Ltrain, Ftrain,option);
%     end
% end
% save('gaussian_rbf.mat','score_rbf');



score_linear = zeros(length(c_vector),1);
for i =1:length(c_vector)
        c=c_vector(i);
        option = ['-s 0 -t 0 -c ' num2str(c)  '  -q'];
        fprintf(option);
        %score_linear(i) = svmtrain(Ltrain, Ftrain,option);
        model = svmtrain(Ltrain, Ftrain,option);
end
%save('linear.mat','score_linear');

% 
% gamma_vector = -5:3;
% score_tanh = zeros(length(c_vector),length(gamma_vector));
% for i =1:length(c_vector)
%     for j = 1:length(gamma_vector)
%         c=c_vector(i);
%         gamma = gamma_vector(j);
%         option = ['-s 0 -t 3 -c ' num2str(c) ' -r 1 -g ' num2str(gamma) ' -v 10 -q'];
%         fprintf(option);
%         score_tanh(i,j) = svmtrain(Ltrain, Ftrain,option);
%     end
% end
% save('tanh.mat','score_tanh');
fake_label = rand(500,1);
[Ptest] = svmpredict(fake_label, Ftest, model);
name = 'songyaos';
save('imagetest.mat','Ptest','name');
webpageDisplay(Itest,Ptest,C);

