function A = krp_Rd(F)

% KRP_RD   R-D Khathri-Rao (column-wise Kronecker) product
%
%  A = KRP_RD(F) computes the Khathri-Rao product of all matrices given in
%  the length-R cell-array F, i.e., A = KRP(KRP(KRP(...,F{R}),F{R-1}),...,F{2}),F{1})
%  All matrices in F are required to have the same number of columns.
%
% Author:
%    Florian Roemer, Communications Resarch Lab, TU Ilmenau
% Date:
%    Dec 2007

R = length(F);
A = F{1};
for r = 2:R
    A = krp(A,F{r});
end