function err = comp_facerr(Fhat,Fref)
% COMP_FACERR   Compute MSE between loading matrices correcting permutation
%               and scaling ambiguities
%
% Syntax:
%    e = COMP_FACERR(F1,F2)
%
% where F1 and F2 are matrices of same size, e.g., estimates of a loading
% matrix in some mode, or cell arrays of matrices.
% If F1 and F2 are matrices the return parameter e is a scalar, otherwise
% it is a vector containing one MSE for each element of the cell arrays.
% The error is a relative mean square error, i.e., the squared Frobenius
% norm of the error matrix divided by the squared Frobenius norm of F2.
if ~iscell(Fhat)
    Fhat = {Fhat};
    Fref = {Fref};
end
R = length(Fhat);
err = zeros(1,R);
for r = 1:R
    % first, normalize
    Fhat{r} = Fhat{r}*diag(1./sqrt(sum(abs(Fhat{r}).^2,1)));
    Fref{r} = Fref{r}*diag(1./sqrt(sum(abs(Fref{r}).^2,1)));
    p_r = size(Fhat{r},2);
    % second, fix permutation
    fromassign = 1:p_r;
    toassign = 1:p_r;
    order = zeros(1,p_r);
    for k = 1:p_r
        A = abs(Fhat{r}(:,toassign)'*Fref{r}(:,fromassign));
        [m,w] = max(A(:));
        [mx,my] = ind2sub([p_r-k+1,p_r-k+1],w);
        order(fromassign(my)) = toassign(mx);
        toassign = toassign([1:mx-1,mx+1:end]);
        fromassign = fromassign([1:my-1,my+1:end]);
    end
    
    % third, fix phase (+scaling, again):
    alpha = zeros(1,p_r);
    for k = 1:p_r
        alpha(k) = Fhat{r}(:,order(k))'*Fref{r}(:,k);
    end
    Fhat{r} = Fhat{r}(:,order)*diag(alpha);
    
    
    % finally, output error
    err(r) = norm(Fhat{r}(:)-Fref{r}(:))^2/norm(Fref{r}(:))^2;
end