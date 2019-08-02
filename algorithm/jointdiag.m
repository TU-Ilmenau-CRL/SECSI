function [X,Q_total,err_iter] = jointdiag(X, options)

% JOINTDIAG    Joint diagonalization of N real-valued matrices
%
%  U = JOINTDIAG(U) jointly diagonalizes N matrices which should
%  be provided in U as a cell array of length N. The
%  matrices are expected to be square and of equal size.
%  The output argument is a cell array of length N that contains
%  the result of the diagonalization.
%  U = JOINTDIAG(U,OPTIONS) allows to control some of the options
%  The OPTIONS parameter is expected to be a STRUCT that can include
%  the following fields:
%      'DisplayIter': Display the evolution of the error for the
%             iterations. Value can be 0 or 1. Defaults to 0.
%      'DisplayWarnings': Display a warning when the joint diagonalization
%             did not converge. Value can be 0 or 1. Defaults to 1.
%      'MaxIter': Maximum number of iterations allowed. Defaults to 50.
%
%  Example:
%     U = JOINTDIAG(U,struct('DisplayWarnings',0,'MaxIter',10))
%  limits the algorithm to 10 iterations and switches off warnings.
%
% [U,E] = JOINTDIAG(U) additionally outputs the evolution of the 
% error over the iteration numbers. LENGTH(E)-1 is the actual number
% of iterations needed since E(1) represents the initial error.
%
% This method is due to the publication:
% SIMULTANEOUS DIAGONALIZATION WITH SIMILARITY TRANSFORMATION FOR NON-DEFECTIVE MATRICES
% by Tuo Fu, Xiqi Gao published at the ICASSP 2006.
%
% The implementation was done by Florian Roemer, TU Ilmenau, CRL in Aug 2006.

% Set default params
DEFAULT_MaxIter = 50;
DEFAULT_DisplayWarnings = 0;
DEFAULT_DisplayIter = 0;

threshold = 1e-10;

% Parse input args
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

% Store slices of input in cell array if input is given as a 3-D array
cellmode = 1;
if ~iscell(X)
    if ndims(X) ~= 3
        error('Input should be either a cell array of squared matrices or a 3-dimensional array.');
    end
    cellmode = 0;
    X0 = X;
    X = cell(1,size(X0,3));
    for n = 1:length(X)
        X{n} = X0(:,:,n);
    end
end

% Number of slices
R = length(X);

% Input matrices must be real-valued
for r = 1:R
    if ~isreal(X{r})
        error('This method only works for real-valued matrices.');
    end
end

% Input matrices must be square and of equal size
d = size(X{r},1);
for r = 1:R
    if any(size(X{r},1)~=d)
        error('This method requires all matrices to be square and of equal size.');
    end
end

% Compute residual error for 'Sweep #0' (does not equal squared sum of elements?!?)
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

% Initialize additional output params
if nargout > 1
    Q_total = eye(d);
end
if nargout > 2
    err_iter = zeros(1,MaxIter);
    err_iter(1) = e;
end

%% Joint diagonalization algorithm
for k = 1:MaxIter
    for p = 1:d-1
        for q = p+1:d
            %[p,q]
            max_e = 0;
            h = -1;
            for r = 1:R
                cur_e = abs(X{r}(p,p) - X{r}(q,q));
                if cur_e > max_e
                    h = r;
                    max_e = cur_e;
                end
            end
            allbutpq = [1:p-1,p+1:q-1,q+1:d];
            
            Kh = sum(X{h}(p,allbutpq) .* X{h}(q,allbutpq) - (X{h}(allbutpq,p) .* X{h}(allbutpq,q)).');
            Gh = sum(X{h}(p,allbutpq).^2 + X{h}(q,allbutpq).^2 + (X{h}(allbutpq,p).^2 + X{h}(allbutpq,q).^2).');
            dh  = X{h}(p,p) - X{h}(q,q);
            xih = X{h}(p,q) - X{h}(q,p);

                  
            yk = atanh((Kh - xih*dh)/(2*(dh^2+xih^2)+Gh));
            Sk = eye(d);
            Sk(p,p) = cosh(yk);Sk(p,q) = sinh(yk);
            Sk(q,p) = sinh(yk);Sk(q,q) = cosh(yk);
            Ski = inv(Sk);
            for r = 1:R
                X{r} = Ski*X{r}*Sk;
            end
            
            if nargout>1
                %Q_total = Sk * Q_total;
                Q_total = Q_total * Sk;
            end
            
            xi_ = zeros(1,R);
            d_ = zeros(1,R);
            for r = 1:R
                xi_(r) = -X{r}(q,p) - X{r}(p,q);
                d_(r) = X{r}(p,p) - X{r}(q,q);
            end
            E = 2*sum(xi_ .* d_);
            D = sum(d_.^2 - xi_.^2);
            qt = E / D;
            th1 = atan(qt)/4;
            th2 = atan(qt)/4 + pi/4;
            if cos(4*th1)*D + sin(4*th1)*E > 0
                thetak = th1;
            else
                th2 = atan(qt)/4 + pi/4;
                if cos(4*th2)*D + sin(4*th2)*E > 0
                    thetak = th2;
                else
                    %error('Did not find a solution!');
                end
            end
            
            
            Uk = eye(d);
            Uk(p,p) = cos(thetak);Uk(p,q) = sin(thetak);
            Uk(q,p) = -sin(thetak);Uk(q,q) = cos(thetak);
            
            for r = 1:R
                X{r} = Uk'*X{r}*Uk;
            end
            if nargout>1
                %Q_total = Uk * Q_total;
                Q_total = Q_total * Uk;
            end                  
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
    if DisplayIter
        fprintf('Sweep # %d: e = %g.\n',k,e);
    end
    if nargout > 2
        err_iter(k) = e;
        %err_iter = [err_iter,e];
    end
    if abs(e-olde)<threshold
        break
    end
end

if nargout > 2
    err_iter = err_iter(1:k);
end

if abs(e-olde)>threshold
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