%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Semi-Algebraic CP decomposition via Simultaneous Matrix Diagonalization
%%% (SECSI) - demo script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% written by Florian Roemer, CRL, TU Ilmenau, Jul 2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc
rng('default'); rng(0); %<-- Set seed for reproducibility

addpath algorithm
addpath utils

tic;
fprintf('\nSECSI-Demo\n==========\n');

%% 1) Standard use-case: R-way tensor with rank d is to be decomposed.
% Construct one
M = [6,9,8];R = length(M);
d = 4;
F = cell(1,R);
for r = 1:R
    F{r} = randn(M(r),d)+1i*randn(M(r),d);
end
X0 = cp_construct(F);

% decompose using all-default options
Fhat = SECSI(X0,d);

% Compare result
err = comp_facerr(F, Fhat);
RecErr = relative_error(X0,cp_construct(Fhat));

fprintf('Reconstrunction error no noise: %g\n',RecErr); % no noise -> perfect
fprintf('The error for F_1 is %g\n ', err(1));
fprintf('The error for F_2 is %g\n ', err(2));
fprintf('The error for F_3 is %g\n ', err(3));

%% Typically X0 is contaminated by noise
X = X0 + sqrt(0.1)*(randn(size(X0))+1i*randn(size(X0)))/sqrt(2);

% decompose using all-default options
Fhat = SECSI(X,d);

% Compare result
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X(:))/norm(X(:));
fprintf('Reconstrunction error 10 dB SNR: %g\n',RecErr); % a few percent error


%% Via heuristics, we can control the performance complexity trade-off
% known default heuristics are 'BM', 'REC PS', 'CON PS', 'RES'

fprintf('\nHeuristics:\n');
t1 = toc;Fhat = SECSI(X,d,'BM');t1 = toc - t1;
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X(:))/norm(X(:));
fprintf('  BM: err = %g, time = %g s\n', RecErr, t1);
t1 = toc;Fhat = SECSI(X,d,'REC PS');t1 = toc - t1;
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X(:))/norm(X(:));
fprintf('  REC PS: err = %g, time = %g s\n', RecErr, t1);
t1 = toc;Fhat = SECSI(X,d,'RES');t1 = toc - t1;
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X(:))/norm(X(:));
fprintf('  RES: err = %g, time = %g s\n', RecErr, t1);
t1 = toc;Fhat = SECSI(X,d,'CON PS');t1 = toc - t1;
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X(:))/norm(X(:));
fprintf('  CON PS: err = %g, time = %g s\n', RecErr, t1);

%% However, heuristics can be adapted much more flexibly if desired
% Here is a custom heuristic which considers the best two mode combinations
% according to the CON criterion, then solves the corresponding four SMDs,
% afterwards picks the best two according to the RES criterion and then
% selects the final estimates according to the BM criterion

t1 = toc;
Fhat = SECSI(X,d,[],'whichsmds','bestcond','bestn_cond',2,'selsmds','bestres','bestn_res',2,'selfinal','bm');
t1 = toc - t1;
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X(:))/norm(X(:));
fprintf('  custom: err = %g, time = %g s\n', RecErr, t1);

%% R-way for R>3
% also works, here BM is very slow. Hence the default is REC PS.
fprintf('\n');
M = [5,4,6,5];R = length(M);
d = 3;
F = cell(1,R);
for r = 1:R
    F{r} = randn(M(r),d)+1i*randn(M(r),d);
end
X0 = cp_construct(F);
% decompose using all-default options
Fhat = SECSI(X0,d);

% Compare result
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X0(:))/norm(X0(:));
fprintf('Four-way (no noise): %g\n',RecErr); % no noise -> perfect

%% Special case (1): two-slab, i.e., one of the M(r) is equal to two
% here, a fully closed-form solution exists which is always used.
% heuristics are ignored.
fprintf('\nSpecial cases:\n');
M = [5,2,6,5];R = length(M);
d = 3;
F = cell(1,R);
for r = 1:R
    F{r} = randn(M(r),d)+1i*randn(M(r),d);
end
X0 = cp_construct(F);
X = X0 + sqrt(0.1)*(randn(size(X0))+1i*randn(size(X0)))/sqrt(2);

% decompose using all-default options
Fhat = SECSI(X,d);

% Compare result
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X0(:))/norm(X0(:));
fprintf('  Two-slab (10 dB SNR): %g\n',RecErr); % no noise -> perfect

%% Special case (2): two-component, i.e., d = 2
% here, also a fully closed-form solution exists which is always used.
% again, heuristics are ignored.
M = [4,5,5,6,4,4];R = length(M);
d = 2;
F = cell(1,R);
for r = 1:R
    F{r} = randn(M(r),d)+1i*randn(M(r),d);
end
X0 = cp_construct(F);
X = X0 + sqrt(0.1)*(randn(size(X0))+1i*randn(size(X0)))/sqrt(2);

% decompose using all-default options
Fhat = SECSI(X,d);

% Compare result
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X0(:))/norm(X0(:));
fprintf('  Two-component (10 dB SNR): %g\n',RecErr); % no noise -> perfect


%% Optimal low-rank decompositions
% by default, truncated HOSVD is used, but it can be changed to HOOI if needed
M = [4,5,5,6,4,4];R = length(M);
d = 1;
F = cell(1,R);
for r = 1:R
    F{r} = randn(M(r),d)+1i*randn(M(r),d);
end
X0 = cp_construct(F);
X = X0 + sqrt(0.1)*(randn(size(X0))+1i*randn(size(X0)))/sqrt(2);

% decompose using all-default options
Fhat = SECSI(X,d);

% Compare result
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X0(:))/norm(X0(:));
fprintf('  Rank-One default [trnc. HOSVD] (10 dB SNR): %g\n',RecErr); % no noise -> perfect

% decompose using all-default options
Fhat = SECSI(X,d,[],'usehooi',true);

% Compare result
Xhat = cp_construct(Fhat);
RecErr = norm(Xhat(:)-X0(:))/norm(X0(:));
fprintf('  Rank-One default [HOOI] (10 dB SNR): %g\n',RecErr); % no noise -> perfect