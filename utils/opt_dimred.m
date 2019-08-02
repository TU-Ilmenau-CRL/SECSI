function [U_Cell, T_red, error, norm_max, norm_vol] = opt_dimred(T, R_new, options)
% Computes the best rank - (R1, R2, ..., RN) approximation of a tensor.
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
% |     Last modifications: 2006-06-22
% |     Modified by Arpita Thakre on 2008-09-09
% |     Modified by Florian Roemer on 2009-03-25
% |----------------------------------------------------------------
%
% Ref: 
% L. De Lathauwer, B. De Moor, J. Vandewalle, "On the Best
% Rank-1 and Rank-(R1,R2,...,RN) Approximation of Higher-Order
% Tensors", SIAM J. Matrix Anal. Appl., Vol. 21, No. 4, April 2000,
% pp. 1324--1342.
%
% L. De Lathauwer, J. Vandewalle, "Dimensionality Reduction 
% in Higher-Order Signal Processing and Rank-(R1,R2,...,RN) 
% Reduction in Multilinear Algebra", Lin. Alg. Appl., Special
% Issue Linear Algebra in Signal and Image Processing, 
% Vol. 391, Nov. 2004, pp. 31--55.
%
% Acknowledgement:
%  This script is based on dimred3u, dimred4u, kindly provided to us by
%  Lieven de Lathauwer in May 2006
%
% [U_Cell, T_red, error, norm_max, normevol] = opt_dimred(T, R_new)
%
% calculates the R_new n - rank approximation of T by 
% the higher order orthogonal iteration as presented in the references 
% above. This algorithm leads to the best rank - (R1, R2, ..., RN)
% approximation of T in a least mean square sense. Therefore, the function 
% f(U1, U2, ..., UN) = || T x1 U1' x2 U2' x3... xN UN'|| = || T_red || is 
% maximized. If R_new is only scalar, then all dimensions of 
% T are reduced to the same rank R_new.
%
% With [U_Cell, T_red, error, norm_max, norm_vol] = opt_dimred(T, R_new, options)
% you can control the iteration by the optional parameter options.
% Therefor the iteration is continued as long as 
% || U1^(n+1) - U1^(n) || < options(1) holds, and the maximum number of
% iteration steps in options(2) is not reached. Default values are
% options(1) = 2e-5; and options(2) = 500;
%
% Inputs:  T       - tensor
%          R_new   - vector of new n-ranks
%          options - iteration control vector (optional)
%
% Outputs: U_Cell   - cell array
%          T_red    - rank R_new approximation of T
%          error    - || T - T_red ||
%          norm_max - f(U1,U2,U3,U4) after the iteration
%          morm_vol - evolution of f

% get dimensions
sizes = size(T);
dimension = length(sizes);

% make sure that R is complete
if length(R_new) == 1
    R_new = R_new .* ones(1, dimension);
end
R_new(R_new>sizes) = sizes(R_new>sizes);

% set default options
if nargin == 2
    options = [2e-5 500];
end

% get all unfoldings
A_Cell = unfoldings(T);

% get hosvd
for n = 1:dimension
    [temp1, temp2, U_Cell{n}] = svd(A_Cell{n}', 0);
end

% init new basevectors
for n = 1:dimension
    U_Cell{n} = U_Cell{n}(:, 1:R_new(n)); 
end

% compute new core tensor 
S = core_tensor(T, U_Cell);

% init U_Cell_Old
U1_Old = zeros(sizes(1), R_new(1));
U1_New = U_Cell{1};

% init loop
iter=1;
norm_vol(:,iter) = norm(S(:), 'fro') .* ones(dimension, 1);

% loop

while norm(abs(U1_Old) - abs(U1_New),'fro') > options(1) & iter < options(2)
    
    iter = iter + 1;
    
    U1_Old = U1_New;
    
    for n = 1:dimension
        
        % iteration step
        temp_tensor = T;
                
        for k = 1:dimension
            
            if k == n
                continue
            end
            
            temp_tensor = nmode_product(temp_tensor, U_Cell{k}', k);
            
        end
               
        % maximize norm
        [temp1, temp2, U_newt] = svd( unfolding(temp_tensor, n)', 0 );
        
        % update U_Cell
        U_Cell{n} = U_newt(:,1:R_new(n));
        
        if n == 1
            U1_New = U_newt(:,1:R_new(1));
        end
        
        % calculate core tensor
        S = core_tensor(T, U_Cell);
        
        % get new Frobenius - norm
        norm_vol(n,iter) = norm(S(:), 'fro');

    end
    
end

norm_max = norm_vol(dimension, iter);
T_red = reconstruct(S, U_Cell);
error = ho_norm(T-T_red);
