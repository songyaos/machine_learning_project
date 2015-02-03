function phi= foo2( X,Mu,s )
%construct matrix for gaussian basis function case
%   dataset, center and variance
if size(X,2) ~= size(Mu,2)
    sprintf('error, column not matching');
end
phi_non = exp(-dist2(X,Mu)/2/s^2);
phi = [ones(size(X,1),1), phi_non];




end

