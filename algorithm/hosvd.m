function [S, U_Cell, SD_Cell] = hosvd(T,Ndim)
% Computes the higher order singular value decomposition of a tensor.
%
% |----------------------------------------------------------------
% | (C) 2006 TU Ilmenau, Communications Research Laboratory
% |
% |     Martin Weis
% |     Florian Roemer[
% |     
% |     Advisors:
% |        Dipl.-Ing. Giovanni Del Galdo
% |        Univ. Prof. Dr.-Ing. Martin Haardt
% |
% |     Last modifications: 04.24.2007
% |----------------------------------------------------------------
%
% [S, U_Cell, SD_Cell] = hosvd(T[,Ndim])
%
% computes the higher order singular value decomposition of T, such
% that T = S x1 U_Cell{1} x2 U_Cell{2} x3 ... xN U_Cell{N}; 
% Thereby N is the dimension of tensor T, and xn denotes the 
% n - mode product (type 'help nmode_product' for further informations). 
% All matrices U_Cell{n} are orthogonal.
% The core tensor S is of same size as T, and has the property of
% all - orthogonality. This means that the scalar product of two arbitrary
% subtensors of S is zero. SD_Cell is a cell array of n vectors containing
% the singular values of T.
%
% Input:   T       - tensor
%          Ndim    - number of dimensions. Ndim is optional and 
%             defaults to NDIMS(T). It is useful to specify Ndim when T may
%             have trailing singleton dimensions. For example, a tensor of
%             size 5 x 3 x 4 x 1 cannot be distinguished from a tensor of
%             size 5 x 3 x 4 since Matlab ignores trailing singleton
%             dimensions. NDIMS would therefore yield 3 even though 4
%             dimensions may be needed.
% 
% Outputs: S       - core tensor
%          U_Cell  - cell array with matrices of eigenvectors
%          SD_Cell - cell array with vectors of singular values

% get dimensions
sizes = size(T);
dimension = length(sizes);

if nargin > 1
    if dimension < Ndim
        sizes = [sizes,ones(1,Ndim-dimension)];
        dimension = Ndim;
    end
end

% get unfoldings
A_Cell = unfoldings(T);

% compute svd's of unfoldings
for n = 1:dimension
    [temp, SD_Cell{n}, U_Cell{n}] = svd(A_Cell{n}', 0);
end

% diagonalize SD_Cell
for n = 1:dimension
    SD_Cell{n} = diag(SD_Cell{n});
end

% compute core tensor
S = nmode_product(T, U_Cell{1}', 1); 
for n = 2:dimension
    S = nmode_product(S, U_Cell{n}', n); 
end
