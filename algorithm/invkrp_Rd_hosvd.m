function Factors = invkrp_Rd_hosvd(X,M,usehooi)
% Factors = invkrp_Rd_hosvd(X,M,usehooi)
%
% invert an R-fold Khatri-Rao product. Tries to decompose a given matrix X
% into F1 \krp F2 \krp ... \krp FR, where size(Fr) = [M(r),N] and size(X) =
% [prod(M),N].
%
% Input
%   X - tensor (double)
%   M - size of dimensions
%   usehooi - true / {false}: Use HOOI for optimal truncated Tucker?
%
% Output
%   Factors as cell array of length R, where Factors{r} is M(r) x N
%
% Author: Florian Roemer, Mar 2009
%
% Header added by Mikus Grasis, CRL, August 2018

if prod(M) ~= size(X,1)
    error('X should be of size [PROD(M),N]!');
end
if nargin < 3
    usehooi = false;
end


N = size(X,2);

R = length(M);
Factors = cell(1,R);
if R == 1
    Factors{1} = X;
else
    if any(isnan(X))
        for r = 1:R
            Factors{r} = nan(M(r),N);
        end
    else
        for r = 1:R
            Factors{r} = zeros(M(r),N);
        end
        
        for n = 1:N
            Xn = X(:,n);
            Xn_t = reshape(Xn,M(end:-1:1));
            [S,U,SD] = hosvd(Xn_t); %#ok<ASGLU>
            if usehooi && (R>2)
                [Uc,Xn_t] = opt_dimred(Xn_t,1);
                Sc = core_tensor(Xn_t,Uc);
            else
                [Sc,Uc] = cuthosvd(S,U,1);
            end
            Uc = Uc(end:-1:1);
            for r = 1:R
                Factors{r}(:,n) = Uc{r};
            end
            Factors{1}(:,n) = Factors{1}(:,n) * Sc;
        end        
    end
end