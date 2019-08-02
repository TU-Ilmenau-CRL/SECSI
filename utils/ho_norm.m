function out = ho_norm(T)
% Computes the Frobenius norm of a tensor.
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
% out = ho_norm(T)
%
% computes the Frobenius norm of the tensor T.
%
% Input: T- Tensor
%
% Output: out - Frobenius norm of tensor T

% compute Frobenius - norm
out = sqrt(scalar_product(T, conj(T)));
%out = norm(T(:));