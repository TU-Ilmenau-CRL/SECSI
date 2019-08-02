function T = iunfolding(Tn, n, sizes, order)
% Reconstructs a tensor out of its n'th unfolding.
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
% |     Last modifications: 04.23.2007
% |----------------------------------------------------------------
%
% T = iunfolding(Tn, n, sizes, order)
%
% reproduces the origin tensor T out of its n'th unfolding
% Matrix Tn, produced by the function unfolding (type 'help unfolding' 
% to get additional informations). Therefore, the dimensions of the 
% origin tensor have to be given by the vector sizes. The optional
% parameter order is the ordering used by the unfolding command. If order
% is not given, the function assumes order = 3 (Lathauwer unfolding).
%
% Inputs: Tn    - matrix with the n - mode vectors of a tensor T
%         n     - dimension
%         sizes - vector containg the size of T
%         order - defines the ordering of the n - mode vectors (optional)
% 
% Output: T     - reproduced tensor

% get dimension
dimension = length(sizes);

% make singletons at the end of T possible
if n > dimension
    sizes = [sizes, ones(1, n-dimension)];
    dimension = n;
end

% Set standard Lathauwer unfolding
if nargin == 3
    order = 3;
end

% get permutation vector
switch order
    case 1
        permute_vec = [n, 1:(n-1), (n+1):dimension]; % indices go faster with increasing index
        [temp, ipermute_vec] = sort(permute_vec);
    case 2
        permute_vec = [n, fliplr( [1:(n-1), (n+1):dimension] )]; % indices go slower with increasing index
        [temp, ipermute_vec] = sort(permute_vec);
    case 3
        permute_vec = [n, fliplr( 1:(n-1) ), fliplr( (n+1):dimension )]; % Lathauwer: indices go slower with I_n+1, ... I_N, I_1, ... I_n-1 
        [temp, ipermute_vec] = sort(permute_vec);
    case 4
        permute_vec = [n, fliplr( [ fliplr( 1:(n-1) ), fliplr( (n+1):dimension ) ])]; % flipped Lathauwer
        [temp, ipermute_vec] = sort(permute_vec);
    otherwise
        disp('Error: unknown ordering for n--mode vectors');
        return
end

% get origin tensor
T = permute(reshape(Tn, sizes(permute_vec)), ipermute_vec);
