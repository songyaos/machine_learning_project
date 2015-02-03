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
%%
% phi_d1 = [ones(training_size,1)   x_training];
% phi_d2 = [phi_d1 x_training.^2 ];
% phi_d3 = [phi_d2 x_training.^3 ];
% phi_d4 = [phi_d3 x_training.^4 ];
% phi_d5 = [phi_d4 x_training.^5 ];
% phi_d6 = [phi_d5 x_training.^6 ];
% phi_d7 = [phi_d6 x_training.^7 ];
basis  = 'polynomial';
phi_d1 = designMatrix(x_training , basis , 1);
phi_d2 = designMatrix(x_training , basis, 2);
phi_d3 = designMatrix(x_training , basis, 3);
phi_d4 = designMatrix(x_training , basis, 4);
phi_d5 = designMatrix(x_training , basis, 5);
phi_d6 = designMatrix(x_training , basis, 6);
phi_d7 = designMatrix(x_training , basis, 7);
w_ml_d1 = pinv( (phi_d1') * phi_d1) * phi_d1' * t_training;
w_ml_d2 = pinv( (phi_d2') * phi_d2) * phi_d2' * t_training;
w_ml_d3 = pinv( (phi_d3') * phi_d3) * phi_d3' * t_training;
w_ml_d4 = pinv( (phi_d4') * phi_d4) * phi_d4' * t_training;
w_ml_d5 = pinv( (phi_d5') * phi_d5) * phi_d5' * t_training;
w_ml_d6 = pinv( (phi_d6') * phi_d6) * phi_d6' * t_training;
w_ml_d7 = pinv( (phi_d7') * phi_d7) * phi_d7' * t_training;

y_d1 = phi_d1 * w_ml_d1;
y_d2 = phi_d2 * w_ml_d2;
y_d3 = phi_d3 * w_ml_d3;
y_d4 = phi_d4 * w_ml_d4;
y_d5 = phi_d5 * w_ml_d5;
y_d6 = phi_d6 * w_ml_d6;
y_d7 = phi_d7 * w_ml_d7;

training_e1 = sqrt(1/training_size * sum( (y_d1 - t_training).^2  )  );
training_e2 = sqrt(1/training_size * sum( (y_d2 - t_training).^2  )  );
training_e3 = sqrt(1/training_size * sum( (y_d3 - t_training).^2  )  );
training_e4 = sqrt(1/training_size * sum( (y_d4 - t_training).^2  )  );
training_e5 = sqrt(1/training_size * sum( (y_d5 - t_training).^2  )  );
training_e6 = sqrt(1/training_size * sum( (y_d6 - t_training).^2  )  );
training_e7 = sqrt(1/training_size * sum( (y_d7 - t_training).^2  )  );

training_e = [training_e1 training_e2 training_e3 ...
    training_e4 training_e5 training_e6 training_e7];

phi_d1 = [ones(test_size,1)   x_test];
phi_d2 = [phi_d1 x_test.^2 ];
phi_d3 = [phi_d2 x_test.^3 ];
phi_d4 = [phi_d3 x_test.^4 ];
phi_d5 = [phi_d4 x_test.^5 ];
phi_d6 = [phi_d5 x_test.^6 ];
phi_d7 = [phi_d6 x_test.^7 ];

testy_d1 = phi_d1 * w_ml_d1;
testy_d2 = phi_d2 * w_ml_d2;
testy_d3 = phi_d3 * w_ml_d3;
testy_d4 = phi_d4 * w_ml_d4;
testy_d5 = phi_d5 * w_ml_d5;
testy_d6 = phi_d6 * w_ml_d6;
testy_d7 = phi_d7 * w_ml_d7;

test_e1 = sqrt(1/test_size * sum( (testy_d1 - t_test).^2  )  );
test_e2 = sqrt(1/test_size * sum( (testy_d2 - t_test).^2  )  );
test_e3 = sqrt(1/test_size * sum( (testy_d3 - t_test).^2  )  );
test_e4 = sqrt(1/test_size * sum( (testy_d4 - t_test).^2  )  );
test_e5 = sqrt(1/test_size * sum( (testy_d5 - t_test).^2  )  );
test_e6 = sqrt(1/test_size * sum( (testy_d6 - t_test).^2  )  );
test_e7 = sqrt(1/test_size * sum( (testy_d7 - t_test).^2  )  );

%%
testing_e = [test_e1 test_e2 test_e3 ...
    test_e4 test_e5 test_e6 test_e7];

figure;hold on;
set(gca,'FontSize',20);
plot( 1:7,  training_e, 'ro-'); 
plot (1:7,  testing_e, 'b*-'); 
hold off;
xlabel('polynomial degree');
ylabel('error');
h_legend = legend('training error',  'test error', 2 );

figure;
set(gca,'FontSize',20);
bar(1:12, [ w_ml_d1(2:end)'   w_ml_d1(1,1) ]);
xlabel('11 features and the bias term');
ylabel('coefficients value');
