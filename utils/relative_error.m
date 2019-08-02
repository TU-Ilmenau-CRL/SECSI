function relErr = relative_error(X,Xhat)
% Compute relative error. Input tensors must be of same type, e.g., tensor,
% ktensor, sptensor, or multidim-array...
%
%   RE = ||X-Xhat|| / ||X||
%
% Input
%   X - original tensor
%   Xhat - approximated tensor
% Output
% 	relErr - relative error
% Mikus Grasis, CRL, July 2018

if isa(X,'double')
    % MATLAB
    relErr = norm(Xhat(:)-X(:))./norm(X(:));
else
    % Tensor Toolbox
    relErr = norm(Xhat-X)./norm(X);
end




end