function S = core_tensor(T, U_Cell)
% Calculates the core tensor.
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
% |     Last modifications: 06.22.2006
% |----------------------------------------------------------------
%
% S = reconstruct(T, U_Cell)
%
% computes the core Tensor S by evaluating the equation:
% S = T x1 U_Cell{1}' x2 U_Cell{2}' x3 ... xN U_Cell{N}'; 
% is computed.
%
% Type 'help hosvd' for further Informations.
%
% Inputs: T      - tensor
%         U_Cell - cell array with matrices of eigenvectors
%          
% Output: S      - core tensor

% get dimension
dimension = length(U_Cell);

% compute Tensor
S = nmode_product(T, U_Cell{1}', 1);
for n = 2:dimension
    S = nmode_product(S, U_Cell{n}', n);
end
