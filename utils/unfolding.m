function out = unfolding(T, n, order)
% Computes the n'th Unfolding of a Tensor.
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
% out = unfolding(T, n)
%
% computes a matrix out that contains all n - mode vectors 
% of the given tensor T, with an ordering as defined by de Lathauwer.
%
% out = unfolding(T, n, order)
%
% computes the matrix of n - mode vectors with the following order:
%
%   order = 1: indices of n - mode vectors go faster with increasing index
%   order = 2: indices of n - mode vectors go slower with increasing index
%   order = 3: de Lathauwer: indices go slower with I_n+1, ... I_N, I_1, ... I_n-1
%   order = 4: flipped de Lathauwer ordering
%
% Inputs: T     - tensor
%         n     - dimension
%         order - defines the ordering of the n - mode vectors (optional, defaults to 3)
%         
% Output: out - n'th matrix unfolding of T

% get dimensions
sizes = size(T);
dimension = length(sizes);

% make singletons at the end of T possible
if n > dimension
    sizes = [sizes, ones(1, n-dimension)];
    dimension = n;
end

% Set standard Lathauwer unfolding
if nargin == 2
    order = 3;
end

% permute tensor T for reshape - command
switch order
    case 1
        T = permute(T, [n, 1:(n-1), (n+1):dimension]); % indices go faster with increasing index
    case 2
        T = permute(T, [n, fliplr( [1:(n-1), (n+1):dimension] )] ); % indices go slower with increasing index
    case 3
        T = permute(T, [n, fliplr( 1:(n-1) ), fliplr( (n+1):dimension )]); % Lathauwer: indices go slower with I_n+1, ... I_N, I_1, ... I_n-1 
    case 4
        T = permute(T, [n, fliplr( [ fliplr( 1:(n-1) ), fliplr( (n+1):dimension ) ])]); % flipped Lathauwer
    otherwise
        disp('Error: unknown ordering for n--mode vectors');
        return
end

% compute n'th unfolding
out = reshape(T, [sizes(n), prod(sizes)./sizes(n)]);
