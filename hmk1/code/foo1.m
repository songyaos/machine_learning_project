function Phi = foo1( X, k )
%construct the design matrix Phi from input data X and polynomial degree k
%formula used is in the slide.
[m n]  = size(X);
Phi = ones(m,1);
i=1;
while i <= k
    Phi = [Phi X.^i ];
    i = i+1;
end


end

