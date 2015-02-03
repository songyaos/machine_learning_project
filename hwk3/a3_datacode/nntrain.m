% 
% Train a neural network.
%
% Question 5

% Load data.
% Loads X (28x28x10000): images, t (10000x1): labels
% Note that original labels are integers in 0-9.
load digits10000
%dbstop in nntrain at 81;
%dbstop in nntrain at 49;
K = 10;  % Number of classes.
ETA = 0.1; % Step size for stochastic gradient descent.
MAX_ITER = 20; % Maximum number of iterations through the training data.

% Transform digits to 10000x784, remove spatial structure.
Xt = transformDigits(X);
t = t+1; % Encode as 1-10, digit+1.

% Set up training and testing sets.
TRAIN_INDS=1:500;
TEST_INDS=501:1000;
% Use below instead if you wish to use the remainder as test data.
% TEST_INDS=setdiff(1:size(Xt,1),TRAIN_INDS);

Xtest=Xt(TEST_INDS,:);
ttest=t(TEST_INDS);

Xtrain=Xt(TRAIN_INDS,:);
ttrain=t(TRAIN_INDS);
[N D] = size(Xtrain);

% Create neural network data structure.
% Simple version, have weight vector per node, all nodes in a layer are same type.
% NN(i).weights is a matrix of weights, each row corresponds to the weights for a node at the next layer.
%  I.e. a_k = NN(i).weights(k,:) * z', where z is the vector of node outputs at the preceding layer.
H = 500;  % Number of hidden nodes.
clear NN;
NN = struct('weights', rand(D+1,H),'type','sigmoid');
NN(1).weights = eye(D+1,H);
NN(2).weights = 0.1*rand(H+1,K);
NN(2).type = 'softmax';



% Stochastic gradient descent with back-propagation.
% Training/testing set accuracy
tra_all=[];
tea_all=[];
for iter=1:MAX_ITER
  fprintf('Training neural network iter %d/%d: ', iter, MAX_ITER);
  tt = clock;
  for x_i=1:N%training size
    [A,Z] = feedforward(Xtrain(x_i,:),NN);

    % Output layer derivative.
    % Assume classification with softmax.
    % Note: code for multiple hidden layers should use a for loop, but the first/last layers are special cases, hence no loop used here.
    % TO DO:: fill this in.
    %dW2 = zeros(H+1,K);%501*10
    t_current = zeros(1,K);
    t_current(ttrain(x_i)) = 1;%only one entry is 1 
    y_current  = Z{2}; %row vector 1*10
    diff1 = y_current - t_current; %row vectorn or training schemes .... We initialized the biases to be 0 and the weights Wij at each layer with the ...
    z_current = [Z{1} 1];%row vector 1*501
    dW2 = z_current' * diff1; %501*10

    % Hidden layer derivative.
    % Backpropagate error from output layer to hidden layer.
    % TO DO:: fill this in.
    %dW1 = zeros(D+1,H); %785 *500
    x_current = [Xtrain(x_i,:) 1];%row vec tor 1*(784+1)
    g_a = Z{1};%row vector
    g_ad = g_a .*(1-g_a);%row vector 1*500
    diff2 = g_ad .* (NN(2).weights(1:size(Xtrain,1), :) * diff1')'; %row vector 1*500
    dW1 = x_current' * diff2; % 785*500
    % Apply the computed gradients in a stochastic gradient descent update.
    NN(2).weights = NN(2).weights - ETA*dW2;
    NN(1).weights = NN(1).weights - ETA*dW1;
  end

  tra_all(iter) = computeAcc(Xtrain,NN,ttrain); 
  tea_all(iter) = computeAcc(Xtest,NN,ttest);
  fprintf('training accuracy = %.4f, took %.2f seconds\n',tra_all(iter),etime(clock,tt));

end
fprintf('Final test accuracy = %.4f.\n',tea_all(end));


% Set up a figure for plotting training error.
figure(1);
clf;
plot(tra_all,'bo-');
hold on;
plot(tea_all,'ro-');
hold off;
set(gca,'FontSize',15);
xlabel('Iteration');
ylabel('Classification accuracy')
title('Training neural network with backpropagation');
legend('Training set','Test set');
axis([1 MAX_ITER 0 1])


% Produce webpage showing predictions.
fprintf('Producing webpage of results... ');
% Get predictions
[A,Z] = feedforward(Xtest,NN);
% Take max over output layer to get predictions.
[mvals,preds] = max(Z{end},[],2);

% -1 to convert back to actual digits.
webpageDisplay(X,TEST_INDS,preds-1,ttest-1);
fprintf('done.\n  TRY OPENING output.html\n');
