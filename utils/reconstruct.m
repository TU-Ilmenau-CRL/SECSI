function T = reconstruct(S, U_Cell)
% Reconstructs a Tensor out of it's higher order singular value decomposition.
%
% |----------------------------------------------------------------
% | (C) 2006 TU Ilmenau, Communications Research Laboratory
% |
% |     Martin Weis
% |     
% |     Advisors:
% |        Dipl.-Ing. Giovanni Del Galdo
% |        Univ. Prof. Dr.-Ing. Martin Haardt
% |
% |     Last modifications: 06.20.2006
% |----------------------------------------------------------------
%
% T = reconstruct(S, U_Cell)
%
% computes the original Tensor T out of it's higher order singular
% value decomposition. Therefor the equation:
% T = S x1 U_Cell{1} x2 U_Cell{2} x3 ... xN U_Cell{N}; 
% is computed.
% Type 'help hosvd' for further Informations.
%
% Inputs: S      - core tensor
%         U_Cell - cell array with matrices of eigenvectors
%          
% Output: T      - tensor

% get dimension
dimension = length(U_Cell);

% compute Tensor
T = nmode_product(S, U_Cell{1}, 1);
for n = 2:dimension
    T = nmode_product(T, U_Cell{n}, n);
end
