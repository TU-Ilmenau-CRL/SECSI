function Xc = cuttensor(X,p)

% CUTTENSOR   Truncate tensor along all modes.
%
% Syntax:
%    T_c = CUTTENSOR(T,p)
%
% Input:
%    T - tensor or arbitrary size
%    p - integer number specifiying to how many elements T should be
%    truncated along all modes.
% 
% Output:
%    T_c - tensor truncated in such a way that
%      SIZE(T_c,n) = min(p,SIZE(T,n)), i.e., 
%    if the tensor has less elements than p in mode n this mode is not
%    altered.
%
% Notes:
%  For a threedimensional tensor T, CUTTENSOR(T,p) is equivalent to 
%      T(1:min(size(T,1),p),1:min(size(T,2),p),1:min(size(T,3),p))
%
% Examples:
%    SIZE(CUTTENSOR(ZEROS(5,4,7),3)) = [3,3,3]
%    SIZE(CUTTENSOR(ZEROS(5,4,7),4)) = [4,4,4]
%    SIZE(CUTTENSOR(ZEROS(5,4,7),5)) = [5,4,5]
%    SIZE(CUTTENSOR(ZEROS(5,4,7),6)) = [5,4,6]
%
% Author:
%    Florian Roemer, Communications Resarch Lab, TU Ilmenau
% Date:
%    Dec 2007


thesubs = cell(1,ndims(X));
for n = 1:ndims(X)
    thesubs{n} = 1:min(p,size(X,n));
end
    
Xc = subsref(X,struct('type','()','subs',{thesubs}));