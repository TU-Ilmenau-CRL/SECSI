function out = scalar_product(T1, T2)
% Computes the scalar product of 2 tensors.
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
% |     Last modifications: 07.28.2006
% |----------------------------------------------------------------
%
% out = scalar_product(T1, T2)
%
% computes the scalar product of the tensors T1 and T2.
% Therefor the tensors T1 and T2 have to be of the same size.
%
% Inputs: T1 - tensor 1
%         T2 - tensor 2
% 
% Output: out - scalar product of T1 and T2

% get dimension
dimension = length(size(T1));

% check dimension condition
if ~all( size(T1) == size(T2) )
    disp(' ');
    disp('Error, tensors must have same dimensions!');
    disp(' ');
    return
end

% compute scalar product
out = T1.*T2;
for n = 1:dimension
    out = sum(out);
end
