function C = krp(A,B)

% KRP   Khathri-Rao (column-wise Kronecker) product of two matrices.
%
%   C = KRP(A,B) returns the Khathri-Rao product of A and B. The input
%   matrices A and B must have the same number of columns. 
%   For A of size M x P and B of size N x P, the resulting matrix will be
%   of size M*N x P.
%
% Author:
%    Florian Roemer, Communications Resarch Lab, TU Ilmenau
% Date:
%    Dec 2007: Original version
%    Oct 2012: Improved speed 

if size(A,2) ~= size(B,2)
    error('Khathri-Rao product requires two matrices with equal number of columns.');
end

%%% another version using bsx
[M,P] = size(A);N = size(B,1);
C = reshape(bsxfun(@times,reshape(A,[1,M,P]),reshape(B,[N,1,P])),[M*N,P]);
%%% NB in Matlab post-2016, you can simply use this
% C = reshape(reshape(A,[1,M,P]) .* reshape(B,[N,1,P]),[M*N,P]);

%%% fast version w/o repmat
% [M,P] = size(A);N = size(B,1);
% Aind = 1:M;
% Bind = (1:N)';
% C = A(Aind(ones(N,1),:),1:P) .* B(Bind(:,ones(M,1)),1:P);

% %%% fast version w/repmat
% C = A(repmat(1:size(A,1),[size(B,1),1]),1:size(A,2)) .* B(repmat((1:size(B,1)).',[1,size(A,1)]),1:size(B,2));

%%% verbose version
% C = zeros(size(A,1)*size(B,1),size(A,2));
% 
% for n = 1:size(A,2)
%     C(:,n) = kron(A(:,n),B(:,n));
% end