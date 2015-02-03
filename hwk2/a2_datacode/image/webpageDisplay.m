function webpageDisplay(I,P,C)
% Produce a webpage to visualize classification output.
% Creates/overwrites output.html
%
% I: N-by-1 cell array of relative path image URLs.
% P: N-by-1 array of predictions (integers)
% C: K-by-1 cell array of class names.  P must have values in 1,2,...,K.

base_url='http://labelme.csail.mit.edu/Images/users/antonio/static_sun_database';
out_file = 'output.html';


N = size(I,1);
K = size(C,1);

% Check for valid input.
assert(N == size(P,1),'P and I have different numbers of examples');
assert(length(setdiff(P,1:K))==0,'P contains invalid predictions');


% Produce simple webpage
fp = fopen(out_file,'w');
fprintf(fp,'<html>\n');
fprintf(fp,'<table>\n');

% Generate a table row for each image.
for e_i=1:N
  fprintf(fp,'<tr>\n');
  fprintf(fp,'  <td><img src="%s%s" width=400></td>\n',base_url,I{e_i});
  fprintf(fp,'  <td>%s</td>\n',C{P(e_i)});
  fprintf(fp,'</tr>\n');
end



fprintf(fp,'</table>\n');
fprintf(fp,'</html>\n');
fclose(fp);
