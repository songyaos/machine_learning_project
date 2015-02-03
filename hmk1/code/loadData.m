function [t,X] = loadData()
% [t,X] = loadData()
%
%
% Read the mpg dataset
% t is an n-by-1 vector of target values, house prices
% X is n-by-d matrix of input variables, each row is one training sample
%
% The Housing dataset comes from the UCI repository, please see:
% https://archive.ics.uci.edu/ml/datasets/Housing
% for documentation on the dataset, including description of input variables

A = textread('housing.data');

% Select subset of features.
t = A(:,14);
%X = A(:,1:13);
X = A(:,setdiff(1:14,[14 1 12]));

% Randomize rows, there is structure in the ordering of the rows.
% Use a fixed random permutation.
% If interested, see what happens with a real random permuation:
%rp = randperm(size(X,1));
load('rp.mat');  % Get the permutation.
X = X(rp,:);
t = t(rp,:);
%save('rp.mat','rp');


