function [X,Q_total,err_iter] = jointdiag_c(X, options)

% JOINTDIAG_C    Joint diagonalization of N complex-valued matrices
%
%  U = JOINTDIAG_C(U) jointly diagonalizes N matrices which should
%  be provided in U as a cell array of length N. The
%  matrices are expected to be square and of equal size.
%  The output argument is a cell array of length N that contains
%  the result of the diagonalization.
%  U = JOINTDIAG_C(U,OPTIONS) allows to control some of the options
%  The OPTIONS parameter is expected to be a STRUCT that can include
%  the following fields:
%      'DisplayIter': Display the evolution of the error for the
%             iterations. Value can be 0 or 1. Defaults to 0.
%      'DisplayWarnings': Display a warning when the joint diagonalization
%             did not converge. Value can be 0 or 1. Defaults to 1.
%      'MaxIter': Maximum number of iterations allowed. Defaults to 50.
%
%  Alternatively, U can be provided as a 3-dimensional array of size
%  [Q,Q,N] which will be treated as N matrices of size Q by Q. In this case
%  the return value is a matrix of same size.
%
%  Example:
%     U = JOINTDIAG_C(U,struct('DisplayWarnings',0,'MaxIter',10))
%  limits the algorithm to 10 iterations and switches off warnings.
%
%
% [D,Q] = JOINTDIAG_C(U) additionally outputs the transformation matrix,
% such that U{n} = Q * D{n} * inv(Q) for all n.
%
% [U,Q,E] = JOINTDIAG_C(U) additionally outputs the evolution of the 
% error over the iteration numbers. LENGTH(E)-1 is the actual number
% of iterations needed since E(1) represents the initial error.
%
% This method is due to the publications:
% [1] SIMULTANEOUS DIAGONALIZATION WITH SIMILARITY TRANSFORMATION FOR NON-DEFECTIVE MATRICES
% by Tuo Fu, Xiqi Gao published at the ICASSP 2006.
% [2] SIMULTANEOUS DIAGONALIZATION BY SIMILARITY TRANSFORMATION WITH
% APPLICATION TO MULTIDIMENSIONAL HARMONIC RETRIEVAL AND RELATED PROBLEMS
% by Tuo Fu, Xiqi Gao (to be published).
% [3] JACOBI ANGLES FOR SIMULTANEOUS DIAGONALIZATION
% by J.-F. Cardoso and A. Souloumiac in SIAM J. Matrix Anal. Appl., vol. 17
% no 1, pp. 161--164, 1996.
%
% The implementation was done by Florian Roemer, TU Ilmenau, CRL in Mar 2007.

DEFAULT_MaxIter = 50;
DEFAULT_DisplayWarnings = 0;
DEFAULT_DisplayIter = 0;

threshold = 1e-8;

if nargin < 2
    DisplayIter = DEFAULT_DisplayIter;
    MaxIter = DEFAULT_MaxIter;
    DisplayWarnings = DEFAULT_DisplayWarnings;
else
    if isfield(options,'DisplayIter')
        DisplayIter = options.DisplayIter;
    else
        DisplayIter = DEFAULT_DisplayIter;
    end
    if isfield(options,'DisplayWarnings')
        DisplayWarnings = options.DisplayWarnings;
    else
        DisplayWarnings = DEFAULT_DisplayWarnings;
    end
    if isfield(options,'MaxIter')
        MaxIter = options.MaxIter;
    else
        MaxIter = DEFAULT_MaxIter;
    end
end

cellmode = 1;
if ~iscell(X)
    if ndims(X) ~= 3
        error('Input should be either a cell array of square matrices or a 3-dimensional array.');
    end
    cellmode = 0;
    X0 = X;
    X = cell(1,size(X0,3));
    for n = 1:length(X)
        X{n} = X0(:,:,n);
    end
end

R = length(X);


d = size(X{1},1);
for r = 1:R
    if any(size(X{r},1)~=d)
        error('This method requires all matrices to be square and of equal size.');
    end
end

if nargout>1
    Q_total = eye(d);
end


Converged = 0;


e = 0;
for r = 1:R
    for nx = 1:d-1
        for ny = nx+1:d
            e = e + abs(X{r}(nx,ny)).^2;
        end
    end
end

if DisplayIter
    fprintf('Sweep # 0: e = %g.\n',e);
end


if nargout > 2
    err_iter = zeros(1,MaxIter);
    err_iter(1) = e;
    %err_iter = e;
end

for k = 2:MaxIter
    smax = 0;
    for p = 1:d-1
        for q = p+1:d
            %[p,q]
            %max_e = 0;
            %h = -1;
            %for r = 1:R
            %    cur_e = abs(X{r}(p,p) - X{r}(q,q));
            %    if cur_e > max_e
            %        h = r;
            %        max_e = cur_e;
            %    end
            %end
            
            
            %%% hyperbolic rotation: according to Tuo Fu, Xiqi Gao
            allbutpq = [1:p-1,p+1:q-1,q+1:d];
            
            
            sum1 = 0;sum2 = 0;sum3 = 0;
            for r = 1:R
                Cn = X{r}*X{r}' - X{r}'*X{r};
                sum1 = sum1 + Cn(p,q);
            end
                  
            alphak = angle ( sum1 ) - pi/2;
            
            for r = 1:R
                %Kn = sum(X{r}(p,allbutpq) .* X{r}(q,allbutpq) - (X{r}(allbutpq,p) .* X{r}(allbutpq,q)).');
                Gn = sum(abs(X{r}(p,allbutpq)).^2 + abs(X{r}(q,allbutpq)).^2 + (abs(X{r}(allbutpq,p)).^2 + abs(X{r}(allbutpq,q)).^2).');
                %Gn = 0;
                %for nj = 1:d
                %    if (nj ~= p) && (nj ~= q)
                %        Gn = Gn + abs(X{r}(p,nj))^2 + abs(X{r}(q,nj))^2 + abs(X{r}(nj,p))^2 + abs(X{r}(nj,q))^2;
                %    end
                %end
                xin =  exp(j*alphak)*X{r}(q,p) + exp(-j*alphak)*X{r}(p,q);
                dn = X{r}(p,p) - X{r}(q,q);
                sum2 = sum2 + abs(dn)^2 + abs(xin)^2;
                sum3 = sum3 + Gn;
            end

            %yk = atanh((Kh - xih*dh)/(2*(dh^2+xih^2)+Gh));
            yk = atanh(-abs(sum1) / (2*sum2 + sum3));
            
            Sk = eye(d);
            Sk(p,p) = cosh(yk);Sk(p,q) = sinh(yk)*(-j)*exp(j*alphak);
            Sk(q,p) = sinh(yk)*(j)*exp(-j*alphak);Sk(q,q) = cosh(yk);
            Ski = inv(Sk);
            for r = 1:R
                X{r} = Ski*X{r}*Sk;
            end
            
            if nargout>1
                %Q_total = Sk * Q_total;
                Q_total = Q_total * Sk;
            end
            
            smax = max(smax,abs(sinh(yk)));
            
            %%% Jacobi angles: according to Cardoso/Souloumiac
            G = zeros(3,3);
            for r = 1:R
                hXr = [X{r}(p,p) - X{r}(q,q), X{r}(p,q) + X{r}(q,p), j*(X{r}(q,p) - X{r}(p,q))];
                G = G + real(hXr' * hXr);
            end
            [Q,L] = eig(G);
            [L,w] = sort(diag(L));
            % largest eigenvalue: L(3) -> position: w(3)
            xyz = Q(:,w(3));

            %r = norm(xyz,2); % <- not neccessary since matlabs evs are unit norm
            %c = sqrt((xyz(1) + r)/(2*r));
            %s = (xyz(2) - j*xyz(3))/(sqrt(2*r*(xyz(1)+r)));
            if xyz(1)<0 , xyz= -xyz; end ;
            c = sqrt(xyz(1)/2 + 1/2);
            s = 0.5*(xyz(2)-j*xyz(3))/c;
            
            Uk = eye(d);
            %Uk(p,p) = cos(thetak);Uk(p,q) = sin(thetak)*(-exp(j*phik));
            %Uk(q,p) = sin(thetak)*(exp(-j*phik));Uk(q,q) = cos(thetak);
            Uk(p,p) = c;Uk(p,q) = -conj(s);
            Uk(q,p) = s;Uk(q,q) = c;
            
            for r = 1:R
                X{r} = Uk'*X{r}*Uk;
            end
            if nargout>1
                %Q_total = Uk * Q_total;
                Q_total = Q_total * Uk;
            end            
            smax = max(smax,abs(s));
        end
    end
    olde = e;
    e = 0;
    for r = 1:R
        for nx = 1:d-1
            for ny = nx+1:d
                e = e + abs(X{r}(nx,ny)).^2;
            end
        end
    end
    if nargout > 2
        err_iter(k) = e;
        %err_iter = [err_iter,e];
    end
    if DisplayIter
        fprintf('Sweep # %d: e = %g.\n',k,e);
    end
    %if abs(e-olde)<threshold
    if smax<threshold
        Converged = 1;
        break
    end

end

if nargout > 2
    err_iter = err_iter(1:k);
end

if ~Converged
    if DisplayWarnings
        fprintf('Warning: Joint diagonalization did not converge (%g -> %g).\n',olde,e);
    end
end

if ~cellmode
    X0 = X;
    X = zeros(size(X0{1},1),size(X0{1},2),length(X0));
    for n = 1:size(X,3)
        X(:,:,n) = X0{n};
    end
end