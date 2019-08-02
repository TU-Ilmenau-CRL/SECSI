function [F_est,infos] = SECSI(varargin)

% SECSI   Semi-Algebraic CP decomposition via Simultaneous Matrix Diagonalization
%
% SECSI is a framework to compute CP (CANDECOMP/PARAFAC or Canonical
% Polyadic) decompositions based on Simultaneous Matrix Diagonalization,
% i.e., without making any use of Alternating Least Squares. Since for a
% given tensor, not only one, but many SMDs can be constructed, the
% accuracy-complexity trade-off can be flexibly controlled within SECSI. 
%
% The basic usage is: 
%
%      Fhat = SECSI(X,d)
%
% where X is an R-way array of size M(r) along mode r=1, 2, ..., R and d is
% the assumed CP rank, i.e., the model order. The return parameter Fhat is
% a length-R cell array containing the R estimated loading matrices of size
% M(r) x d. Calling SECSI this way will  invoke the default SECSI algorithm
% (see below). This can be changed by adding more parameters:
%
%      Fhat = SECSI(X,d,heuristic)
%
% where heuristic is a string identifying which SECSI algorithm to use.
% Currently known identifiers are:
%
%        'BM' [default for R=3]: Solve *all* SMDs possible, for a
%            non-degenerate tensor we have R*(R-1). Then, test all possible
%            combinations of estimates in an exhaustive manner and pick the
%            one with the smallest reconstruction error compared to the
%            noisy tensor X. This requires to resolve permutation and
%            scaling ambiguities between estimates from different SMDs (see
%            below). For R>3 BM may be very slow.
%         'REC PS' [default for R>3]: As before, solve *all* SMDs possible
%            but then, test only combinations of estimates originating from
%            the same SMD (PS = paired solutions). Often close to BM but a
%            lot faster, especially for R>3.
%         'CON PS': Solve only two SMDs originating from the slices with
%            the minimum conditioning number. Then proceed as in REC PS,
%            i.e., following "paired solutions".
%         'RES': Solve all SMDs possible, then pick one with the smallest
%            residual, i.e., the one where the norm of the residual
%            off-diagonal elements after diagonalization is the smallest.
%            This is the only heuristic which does not compare different
%            estimates in terms of their reconstruction error.
%
% Additionally, SECSI can be adjusted further by passing additional
% optional input arguments as parameter/value pairs, i.e.,
% 
%      Fhat = SECSI(X,d,heuristic,'parameter',value)
%
% Currently supported parameters are:
%           'usehooi' - true / {false}
%               By default, SECSI performs Tucker compression via
%               the truncated HOSVD, which is not the optimal
%               low-n-rank-approximation in the Frobenius norm sense. The
%               HOOI algorithm can be used instead. This enhances the
%               accuracy (very) slightly at the cost of higher
%               computational complexity.
%           'whichsmds' - 'all' / 'bestcond'
%               Overrides default value controlling which SMDs to solve.
%               CON PS uses 'bestcond' whereas BM, REC PS and RES use
%               'all'. By default 'bestcond' orders the possible mode
%               combination by the conditioning of the corresponding pivots
%               and then picks the best. Note that each mode combination
%               gives two SMDs (rhs, lhs).
%           'bestn_cond' - {1}
%               Number of mode combinations to consider when using
%               'whichsmds'='bestcond'. By default, only the best mode
%               combination is used, but setting bestn_cond to a higher
%               value (e.g., 2) allows to consider the best n.
%            'selsmds' - 'all' / 'bestres'
%               Post-selection of SMDs after they have been solved. Here,
%               'all' means no post-selection is applied, i.e., all SMDs
%               are selected. This is the default for 'BM', 'REC PS' and
%               'CON PS'. Only 'RES' uses 'bestres', which means that the
%               SMD with the lowest residual (norm of off-diagonal elements
%               after diagonalization) is selected. 
%            'bestn_res' - {1}
%               As for bestn_cond, bestn_res can be used to control how
%               many SMDs are selected via the 'selsmds'='bestres'
%               criterion. Specifying a value n larger than one corresponds
%               to selecting the best n.
%            'selfinal' - 'bm' / 'ps'
%               Controls how to select the final estimates. Here, 'bm'
%               refers to the exhaustive best matching scheme, considering
%               all possible pairs of estimates, whereas 'ps' limits the
%               search to estimates originating from the same SMD. 
%            'solveperm' - {'cp amp'} / 'ang loadvec'
%               Only relevant for 'selfinal'='bm': to combine estimates
%               from different SMDs, permutation ambiguities between
%               different SMDs need to be resolved. This parameter controls
%               how. Here, 'cp amp' refers to ordering columns according to
%               the magnitude of the "Least-Squares CP amplitudes", which
%               is the default option. On the other hand 'ang loadvec'
%               orders by arranging the estimated loading vectors so that
%               the angular distance between columns is minimized.
% 
% Examples:
%    Fhat = SECSI(X,d)           
%        -> use just the default options
%    Fhat = SECSI(X,d,'REC PS')  
%        -> change heuristic to REC PS
%    Fhat = SECSI(X,d,[],'usehooi',true)  
%        -> keep default settings but switch on HOOI
%    Fhat = SECSI(X,d,[],'whichsmds','bestcond','bestn_cond',2,...
%           'selsmds','bestres','bestn_res',2,'selfinal','bm');
%        -> define a custom heuristic
%    Fhat = SECSI(X,d,'CON PS','bestn_cond',2)
%        -> use CON PS heuristic, only change one setting
%             
% Notes:
%    The philosophy of this implementation is to provide a general purpose
%    implementation of SECSI. It therefore tries to adapt to specific cases
%    while still allowing the user to manually control it. Therefore, it
%    makes as few assumption on the data as possible. It can be real or
%    complex, it can have an arbitrary number of dimensions and no
%    particular order is needed. It detects special cases as two-slab,
%    rank-one, and rank-two, where no SMDs are solved but instead
%    closed-form solutions are directly applied.
%    There are cases where SECSI fails. It tries its best to detect these
%    cases, output a warning, and return NaNs. 
%
% Author:
%    Florian Roemer, CRL, TU Ilmemau, Jul 2011
% 
% References:
%    F. Roemer and M. Haardt, "A semi-algebraic framework for approximate
%    CP decompositions via simultaneous matrix diagonalizations (SECSI),"
%    Signal Processing, vol. 93, no. 9, pp. 2722-2738, 2013.
%% Add dependencies
addpath('utils');
addpath('algorithm');

%% Verify input using input parser
p = inputParser;
p.addRequired('X',@(x) isnumeric(x) && all(isfinite(x(:))));
p.addRequired('d',@(x) isnumeric(x) && isscalar(x) && isfinite(x) && (x>0) && (mod(x,1)==0));
p.addOptional('heuristic',[],@(x) ischar(x) || isempty(x));
p.addParameter('usehooi',false,@islogical);
p.addParameter('overridecf2',false,@islogical);
p.addParameter('whichsmds',[],@(x) isempty(x) || strcmp(x,'all') || strcmp(x,'bestcond'));
p.addParameter('selsmds',[],@(x) isempty(x) || strcmp(x,'all') || strcmp(x,'bestres'));
p.addParameter('selfinal',[],@(x) isempty(x) || strcmp(x,'bm') || strcmp(x,'ps'));
p.addParameter('solveperm',[],@(x) isempty(x) || strcmp(x,'cp amp') || strcmp(x,'ang loadvec'));
p.addParameter('bestn_cond',[],@(x) isempty(x) || isnumeric(x));
p.addParameter('bestn_res',[],@(x) isempty(x) || isnumeric(x));

p.parse(varargin{:});

pars = p.Results;
X = pars.X;
d = pars.d;

warning('on','SECSI:rankdeficiency');

% and more checks
M = size(X);
R = length(M);
if R<3, error('Need at least 3 dimensions.');end
if any(M==1), error('Singleton dimensions are not allowed.');end
infos = struct;

global SECSI_debug;

%% HOOI as a general preprocessing step: do it here if asked for.
if pars.usehooi
    [dummy,X] = opt_dimred(X,d*ones(1,R));  %#ok<ASGLU>
end

%% Special cases
% two-slab
if any(M==2) && (~pars.overridecf2)
    [F_est,rt] = SECSI_cf_twoslab(X,d);
    if ~rt
        warning('SECSI:reliabilitytest','Reliability test of closed-form solution has failed, resulting model is complex.');
    end
    return;
end
% two-component
if d==2 && (~pars.overridecf2)
    [F_est,rt] = SECSI_cf_twocomp(X);
    if ~rt
        warning('SECSI:reliabilitytest','Reliability test of closed-form solution has failed, resulting model is complex.');
    end
    return;
end
% rank-one: kinda meaningless but we want to be robust enough to handle it
if d == 1
    [S,U] = hosvd(X);
    [Sc,Uc] = cuthosvd(S,U,1);
    F_est = Uc;
    F_est{1} = F_est{1}*Sc;
    return;
end

%% Now refine heuristics
if isempty(pars.heuristic)
    if R == 3
        pars.heuristic = 'BM';
    else
        pars.heuristic = 'REC PS';
    end
end
heur = struct;
heur.shortname = pars.heuristic;

switch upper(heur.shortname)
    case 'BM'
        heur.whichsmds = 'all';     % which ones to even solve?
        heur.selsmds = 'all';       % which to select after solving
        heur.selfinal = 'bm';       % how to select final estimate
        heur.solveperm = 'cp amp'; % how to fix permutation ambiguities
                        % (alternative: 'ang loadvec')
    case 'CON PS'
        heur.whichsmds = 'bestcond';
        heur.bestn_cond = 1;          % solve the n best. n = 1: only one.
        heur.selsmds = 'all';
        heur.selfinal = 'ps';       
    case 'REC PS'
        heur.whichsmds = 'all';
        heur.selsmds = 'all';
        heur.selfinal = 'ps';
    case 'RES'
        heur.whichsmds = 'all';
        heur.selsmds = 'bestres';
        heur.bestn_res = 1;           % consider the n best. n = 1: only one.
        heur.selfinal = 'ps';
    case 'DUMMYR'
        heur.whichsmds = 'rand';
        heur.selsmds = 'rand';
    case 'DUMMYF'
        heur.whichsmds = 'first';
        heur.selsmds = 'first';
    case 'GENIE'
        heur.whichsmds = 'all';
        heur.selsmds = 'all';
        heur.selfinal = 'genie';
        if ~isfield(SECSI_debug,'trueF')
            error('Genie heuristic requires to specify true loading matrices in global variable SECSI_debug.trueF.');
        end
    otherwise
        error('Unknown heuristic.');
end
% allow manual override:
if ~isempty(pars.whichsmds), heur.whichsmds = pars.whichsmds; end
if ~isempty(pars.selsmds), heur.selsmds = pars.selsmds; end
if ~isempty(pars.selfinal), heur.selfinal = pars.selfinal; end
if ~isempty(pars.bestn_cond), heur.bestn_cond = pars.bestn_cond; end
if ~isempty(pars.bestn_res), heur.bestn_res = pars.bestn_res; end


%% Tucker core via HoSVD (HOOI done before in case required)
[S,U] = hosvd(X);
[Sc,Uc] = cuthosvd(S,U,d);

%% Step 1: solving SMDs
nkl = 0;
optconds = NaN(1,R*(R-1)/2);
Skl_save = cell(1,R*(R-1)/2);
klvalues = zeros(R*(R-1)/2,2);
pivots = zeros(R*(R-1)/2,1);
% Go thru each mode but the last...
for k = 1:R-1
    % Skip mode if dimension M(k) is smaller than rank we are looking for...
    if M(k) < d
        continue
    end
    
    % Consider all modes after the current one...
    for ell = k+1:R
        % Skip mode if dimension M(ell) is smaller than rank we are looking for...
        if M(ell) < d
            continue
        end
        
        % Compute n-mode product of core tensor and the factors in the remaining modes...
        Skl = Sc;
        modes_notkl = [1:k-1,k+1:ell-1,ell+1:R];
        for r = modes_notkl
            Skl = nmode_product(Skl,Uc{r},r);
        end
        
        % Arrange 'S_kl' in slices of size d x d
        Skl = permute(Skl,[k,ell,modes_notkl]); % keep modes k and ell first...
        Skl = reshape(Skl,[d,d,prod(M(modes_notkl))]); % (does nothing in 3-D case)
              
        % Compute condition numbers for each slice
        nSlices = size(Skl,3);
        conds = zeros(1,nSlices); 
        for n = 1:nSlices
            conds(n) = cond(Skl(:,:,n));
        end
        % Determine slice wih the best conditioning number
        [mincond,p] = min(conds);
        
        % Print warning if 'best' slice is rank-deficient
        if rank(Skl(:,:,p)) < d
            warning('SECSI:rankdeficiency','Rank-deficient mode detected. Skipping (this warning is only displayed once)...');
            warning('off','SECSI:rankdeficiency');
            continue
        end
        
        % Store values to solve SMDs in next step
        nkl = nkl + 1; % number of non rank-deficient n-k-l combinations
        pivots(nkl) = p; % slice wih best conditioning number is later used as pivot
        optconds(nkl) = mincond;        
        Skl_save{nkl} = Skl;
        klvalues(nkl,:) = [k,ell];
    end
end
if nkl==0
    warning('SECSI:couldnotdecompose','Could not decompose the tensor: too many rank deficiencies. Returning NaN.');
    F_est = cell(1,R);
    for r = 1:R
        F_est{r} = nan(M(r),d);
    end
    return
end

%% Now select which SMDs we will actually solve
if strcmp(heur.whichsmds,'all')
    NSMDs = nkl;
elseif strcmp(heur.whichsmds,'rand')
    w = ceil(rand*nkl);
    NSMDs = 1;
    Skl_save = Skl_save(w);
    klvalues = klvalues(w,:);
    pivots = pivots(w);    
elseif strcmp(heur.whichsmds,'first')
    NSMDs = 1;
    Skl_save = Skl_save(1);
    klvalues = klvalues(1,:);
    pivots = pivots(1);        
elseif strcmp(heur.whichsmds,'bestcond')
    [optconds,order] = sort(optconds,'ascend');
    Skl_save = Skl_save(order);
    klvalues = klvalues(order,:);
    pivots = pivots(order);    
    NSMDs = min(nkl,heur.bestn_cond); 
end

% Each SMD provides two estimates (one for all from rhs, one for all from lhs)
resids = NaN(1,2*NSMDs);
F_est = cell(2*NSMDs,R);

%% Solve SMDs by similarity transform
for nSMD = 1:NSMDs
    k = klvalues(nSMD,1);
    ell = klvalues(nSMD,2);
    modes_notkl = [1:k-1,k+1:ell-1,ell+1:R];
    Skl = Skl_save{nSMD};
        
        % Multiply all slices by inverse of one particular slice...
        % (we choose the slice with the best conditioning number)
        Skl_rhs = Skl;
        Skl_lhs = Skl;
        nSlices = size(Skl,3);
        for n = 1:nSlices
            % ...from the right hand side
            Skl_rhs(:,:,n) = Skl_rhs(:,:,n)/Skl(:,:,pivots(nSMD));
            % ...and from the left hand side, respectively
            Skl_lhs(:,:,n) = (Skl(:,:,pivots(nSMD))\Skl_lhs(:,:,n)).';            
        end
        
        % Perform joint EVD
        if isreal(X)
            [Qr,Tk,ek] = jointdiag(Skl_rhs);
            [Ql,Tl,el] = jointdiag(Skl_lhs);
        else
            [Qr,Tk,ek] = jointdiag_c(Skl_rhs);
            [Ql,Tl,el] = jointdiag_c(Skl_lhs);
        end
        resids(2*nSMD-1) = ek(end)/nSlices;
        resids(2*nSMD) = el(end)/nSlices;
        
        %%% Obtain two sets of three estimates (paired solutions)
        
        %%% First estimate: from F_r = U_r^{[s]} * T_r
        F_est{2*nSMD-1,k} = Uc{k}*Tk;
        F_est{2*nSMD,ell} = Uc{ell}*Tl;
        
        
        %%%% Second estimate: from diag(S_kl^{rhs}) and diag(S_kl^{lhs})
        F_krp_lhs = zeros(nSlices,d);
        F_krp_rhs = zeros(nSlices,d);
        for n = 1:nSlices
            F_krp_rhs(n,:) = diag(Qr(:,:,n));
            F_krp_lhs(n,:) = diag(Ql(:,:,n));
        end
        % The diagonal elements of the matrices T_1^{-1) * S_3k^{rhs} * T_1 
        % and T_2^{-1) * S_3k^{lhs} * T_2 provide an estimate for the matrix F^{(3)}
        if R > 3
            decomp_krp_lhs = fliplr( invkrp_Rd_hosvd( F_krp_lhs, fliplr(M(modes_notkl)) ) );
            decomp_krp_rhs = fliplr( invkrp_Rd_hosvd( F_krp_rhs, fliplr(M(modes_notkl)) ) );
        else
            decomp_krp_lhs = {F_krp_lhs};
            decomp_krp_rhs = {F_krp_rhs};
        end
        for n = 1:length(modes_notkl)
            F_est{2*nSMD-1,modes_notkl(n)} = decomp_krp_rhs{n};
            F_est{2*nSMD,modes_notkl(n)} = decomp_krp_lhs{n};
        end
        
        %%% Third estimate: LS fit of missing estimate
        
        % Provided we have F^{(1)} and F^{(3)}, the missing F^{(2)} can be
        % found via a pseudo inverse.
        F_est{2*nSMD-1,ell} = unfolding(X,ell)/(krp_Rd( F_est(2*nSMD-1,[ell+1:R,1:ell-1]) ).');
        F_est{2*nSMD,k} = unfolding(X,k)/(krp_Rd(F_est(2*nSMD,[k+1:R,1:k-1])).');
        % This step could be performed later for all heuristics that use
        % post selection, skipping the ones that are discarded, resulting
        % in a slight speed up. We do not do it here, because the
        % programming overhead would be quite significant.
        
        % Note: this step could be replaced by a LS-Khatri-Rao factorization,
        % which is less computationally expensive
end

%% Post selection: after solving SMDs may skip some, e.g., based on residuals
if strcmp(heur.selsmds,'bestres')
    [resids,order] = sort(resids,'ascend');
    F_est = F_est(order,:);
    NumEst = min(heur.bestn_res,2*NSMDs);
elseif strcmp(heur.selsmds,'first')
    F_est = F_est(1,:);
    NumEst = 1;
elseif strcmp(heur.selsmds,'rand')
    w = ceil(rand*2*NSMDs);
    F_est = F_est(w,:);
    NumEst = 1;
elseif strcmp(heur.selsmds,'all')
    NumEst = 2*NSMDs;
end

F_est = F_est(1:NumEst,:);

%% Now, selection of final estimate
if NumEst > 1
    if strcmp(heur.selfinal,'ps')
        recerr = inf*ones(1,NumEst);
        for nEst = 1:NumEst
            Xrec = cp_reconstruct(F_est(nEst,:));
            recerr(nEst) = norm(Xrec(:)-X(:))^2;        
        end
        [m,w] = min(recerr); %#ok<ASGLU>
        F_est = F_est(w,:);
        
    elseif strcmp(heur.selfinal,'bm')
        % Normalize, permute 
        for nEst = 1:NumEst
            if strcmp(heur.solveperm,'cp amp')
                % Find permutation by ordering components according to LS CP amplitudes
                gammahat = krp_Rd(F_est(nEst,:))\X(:);
                [~,order] = sort(abs(gammahat),'descend');
            end
            for r = 1:R
                F_est{nEst,r} = F_est{nEst,r}*diag(1./sqrt(sum(abs(F_est{nEst,r}).^2,1)));
                if strcmp(heur.solveperm,'cp amp')
                    F_est{nEst,r} = F_est{nEst,r}(:,order);
                end
            end
        end
        
        if strcmp(heur.solveperm,'ang loadvec')
            % Fix permutation via angles between loading vectors. phase
            % ambiguity is left but irrelevant since we use abs()
            for r = 1:R
                for nEst = 2:NumEst
                    % Permute F_est{nEst,r} as F_est{1,r}
                    fromassign = 1:d;
                    toassign = 1:d;
                    order = zeros(1,d);
                    for n = 1:d
                        A = abs(F_est{nEst,r}(:,toassign)'*F_est{1,r}(:,fromassign));
    %                     A = abs(pinv(F_est{nEst,r}(:,toassign))*F_est{1,r}(:,fromassign));
                        [~,w] = max(A(:));
                        [mx,my] = ind2sub([d-n+1,d-n+1],w);
                        order(fromassign(my)) = toassign(mx);    
                        toassign = toassign([1:mx-1,mx+1:end]);
                        fromassign = fromassign([1:my-1,my+1:end]);
                    end
                    F_est{nEst,r} = F_est{nEst,r}(:,order);

                end
            end
        end
        
        % Third, test combinations. Phase ambiguity still irrelevant since
        % we use Least Squares fit for CP amplitudes
        combs = zeros(NumEst^R,R);
        for r = 1:R
            combs(:,r) = repmat(kron((1:NumEst)',ones(NumEst^(R-r),1)),[NumEst^(r-1),1]);
        end        
        NumComb = size(combs,1);
        err_save = zeros(NumComb,1); %%% only for debug
        minerr = inf;
        for nComb = 1:NumComb
            curF = cell(1,R);
            for r = 1:R, curF{r} = F_est{combs(nComb,r),r};end
            [Xrec,gammahat] = cp_reconstruct(curF,X,true);
            err = norm(Xrec(:)-X(:));
            err_save (nComb) = err; %%% only for debug
            if err<minerr
                minerr = err;
                bestF = curF;
                bestgamma = gammahat;
            end
        end
        
        % Final estimate
        F_est = bestF;
        % Put amplitudes into first (arbitrary choice)
        F_est{1} = F_est{1}*diag(bestgamma);
            
        
    elseif strcmp(heur.selfinal,'genie')
        ferr = zeros(NumEst,R);
        for r = 1:R
            for nEst = 1:NumEst
                [ferr(nEst,r),F_est{nEst,r}] = comp_facerr(F_est{nEst,r},SECSI_debug.trueF{r});
            end
        end
        [m,w] = min(ferr); %#ok<ASGLU>
        for r = 1:R
            F_est{1,r} = F_est{w(r),r};
        end
        F_est = F_est(1,:);
        [Xrec,gammahat] = cp_reconstruct(F_est,X,true); %#ok<ASGLU>
        F_est{1} = F_est{1}*diag(gammahat);
        % Recerr not really fair comparison: we have used genie-information
        % to correct perm/scal. maybe we shouldn't do this...
    end
end
        

infos.optconds = optconds;
infos.resids = resids;


%% Helper functions

function [Xrec,gammahat] = cp_reconstruct(Factors,X,LSampfit)
% Reconstruction of tensor from CP model

if nargin < 3
    LSampfit = false;
end
if LSampfit
    % Fitting LS amplitudes: not needed if only factors from the same SMD
    % are combined (this already contains an LS fit) but should be used
    % when combining factors from different estimates
%     gammahat = pinv(krp_Rd(Factors(end:-1:1)))*X(:);
    gammahat = krp_Rd(Factors(end:-1:1))\X(:);
    M = size(X);
    Xrec = iunfolding(Factors{1}*diag(gammahat)*krp_Rd(Factors(2:end)).',1,M);
else
    R = length(Factors);
    M = zeros(1,R);
    for r = 1:R,M(r) = size(Factors{r},1);end
    Xrec = iunfolding(Factors{1}*krp_Rd(Factors(2:end)).',1,M);
end

function [Fhat,rt] = SECSI_cf_twoslab(X,d)

% Closed-Form solution for two slabs, i.e., a tensor which has size two in
% one dimension. This function will permute and reshape such a tensor into
% an M x N x 2 tensor, trying to maximize identifiability and then call the
% 3-D algorithm SECSI_cf_twoslab_3d.

M = size(X);
R = length(M);
w = find(M==2);  % find dimension where two slaps exist
w2 = w(1);       % might be multiple, pick first

if R == 3        % 3-way is easier: only permutation
    dims = [1:w2-1,w2+1:R,w2];          % permute size-2 to the end-
    dims_i = [1:w2-1,R,w2:R-1];         % inverse permutation
    X = permute(X,dims);                % permute tensor
    [Fhat_3d,rt] = SECSI_cf_twoslab_3d(X,d); % call 3-D
    Fhat = Fhat_3d(dims_i);             % apply inverse permutation
else
    % R-way is a bit harder: there are multiple ways of reducing to 3-way.
    % The result is an M x N x 2 tensor and identifiability is limited by
    % d<=min(M,N). Therefore, the goal is to reduce the R-way to a 3-way
    % tensor maximizing M and N. This is a combinatorial problem, so
    % instead of going through all combinations we pick the better of two
    % heuristics : 
    % (1) In each step we joint the two smallest dimensions until only 3
    % are left.
    % (2) assign the largest dimension to the first group, the second largest
    % to the second group, the third again to the first group, and so on.
    % Then join all modes from the first group into the new first
    % dimension and all modes from the second group into the new second
    % dimension.
    
    % Heuristic (1)
    % all modes we can join (all except the two-slab mode)
    modes = num2cell([1:w2-1,w2+1:R]);
    sizes = M([1:w2-1,w2+1:R]);
    while length(modes)>2
        % order by size
        [Ms,w] = sort(sizes,'ascend'); %#ok<ASGLU>
        % pick smallest two and join
        modes{w(1)} = [modes{w(1)},modes{w(2)}];
        sizes(w(1)) = sizes(w(1))*sizes(w(2));
        modes(w(2)) = [];
        sizes(w(2)) = [];
    end
    maxd = min(sizes);
    % Heuristic (2)
    joinablemodes = [1:w2-1,w2+1:R]; 
    [Ms,w] = sort(M(joinablemodes),'descend'); %#ok<ASGLU>
    % assign to first half (fh) and second half (sh)
    modes2 = {joinablemodes(w(1:2:end)),joinablemodes(w(2:2:end))};
    sizes2 = [prod(M(modes2{1})), prod(M(modes2{2}))];
    maxd2 = min(sizes2);
    if maxd>maxd2    
        % assign to first half (fh) and second half (sh)
        dims_fh = modes{1};
        dims_sh = modes{2};
    else
        % assign to first half (fh) and second half (sh)
        dims_fh = modes2{1};
        dims_sh = modes2{2};
    end
    % permutation of dimensions
    dims = [dims_fh,dims_sh,w2];
    % inverse permutation
    dims_i = zeros(1,R);dims_i(dims) = 1:R;
    X = permute(X,dims);
    % reshape R-way to 3-way
    X = reshape(X,[prod(M(dims_fh)),prod(M(dims_sh)),M(w2)]);
    % call 3-way solution
    [Fhat_3d,rt] = SECSI_cf_twoslab_3d(X,d);

    % break dimensions back up using LSKRF
    Fhat_fh = fliplr(invkrp_Rd_hosvd(Fhat_3d{1},fliplr(M(dims_fh))));
    Fhat_sh = fliplr(invkrp_Rd_hosvd(Fhat_3d{2},fliplr(M(dims_sh))));
    % all R loading matrices
    Fhat = [Fhat_fh,Fhat_sh, Fhat_3d(3)];
    % undo permutation
    Fhat = Fhat(dims_i);
end

function [Fhat,rt] = SECSI_cf_twoslab_3d(X,d)

% Closed-Form solution for a 3-way tensor with two slabs, expects X to be
% of size M x N x 2 

if ~isreal(X)
    rt = true;
end

% Tucker compression -> d x d x 2
M = size(X);
tuckermodes = M(1:2)>d;
if tuckermodes(1)
    [Uc1,dummy,dummy2] = svd(unfolding(X,1));Uc1 = Uc1(:,1:min(size(X,1),d)); %#ok<ASGLU>
    X = nmode_product(X,Uc1',1);
end
if tuckermodes(2)
    [Uc2,dummy,dummy2] = svd(unfolding(X,2));Uc2 = Uc2(:,1:min(size(X,2),d)); %#ok<ASGLU>
    X = nmode_product(X,Uc2',2);
end

% both slices are T1*diag(...)*T2' so T1 are eigenvectors of X1\X2 or X2\X1
% one of them rank-deficient (e.g., zeros on diagonal) => flip
flipit = false;
if rank(X(:,:,2)) < d
    X = X(:,:,[2,1]);
    flipit = true;
end
if flipit && (rank(X(:,:,2)) < d)
    warning('SECSI:couldnotdecompose','Cannot decompose: Tensor is rank deficient. Returning NaN.');
    F1 = nan(M(1),d);
    F2 = nan(M(2),d);
    F3 = nan(2,d);
    rt = true;
else
    [F1,dummy] = eig(X(:,:,1)/(X(:,:,2)));  %#ok<ASGLU>
    [F2,dummy] = eig((X(:,:,2)\(X(:,:,1))).');  %#ok<ASGLU>
    if isreal(X)
        rt = isreal(F1) && isreal(F2);
    end
    D1 = inv(F1)*X(:,:,1)*inv(F2.'); %#ok<MINV>
    % F1 and F2 may be permuted differently -> D1 permuted diagonal
    [dummy,p] = max(abs(D1)); %#ok<ASGLU>
    % if so, fix
    if any(p~=1:d)
        F1 = F1(:,p);
        D1 = D1(p,:);
    end
    % reconstruct T3 from diagonal
    D2 = inv(F1)*X(:,:,2)*inv(F2.'); %#ok<MINV>
    F3 = [diag(D1).';diag(D2).'];
    if flipit,F3 = F3([2,1],:);end
    % Tucker reconstruction
    if tuckermodes(1),F1 = Uc1*F1;end
    if tuckermodes(2),F2 = Uc2*F2;end
end
Fhat = {F1,F2,F3};

function [Fhat,rt] = SECSI_cf_twocomp(X)

% Closed-Form solution for two-component CP (d=2)

% Tucker "compression" first
[S,U] = hosvd(X);
[Sc,Uc] = cuthosvd(S,U,2);
% CP of core tensor reveals transform matrices, core tensor is 2x2x...x2,
% therefore two-slab solution applies
[That,rt] = SECSI_cf_twoslab(Sc,2);

R = length(That);
Fhat = cell(1,R);
for r = 1:R
    % Tucker reconstruction
    Fhat{r} = Uc{r}*That{r};
end

function [err,Fhat] = comp_facerr(Fhat,Fref)

% COMP_FACERR   Compute MSE between loading matrices correcting permutation
%               and scaling ambiguities
%
% Syntax:
%    e = COMP_FACERR(F1,F2)
%
% where F1 and F2 are matrices of same size, e.g., estimates of a loading
% matrix in some mode, or cell arrays of matrices.
% If F1 and F2 are matrices the return parameter e is a scalar, otherwise
% it is a vector containing one MSE for each element of the cell arrays.
% The error is a relative mean square error, i.e., the squared Frobenius
% norm of the error matrix divided by the squared Frobenius norm of F2.

nc = false;
if ~iscell(Fhat)
    Fhat = {Fhat};
    Fref = {Fref};
    nc = true;
end
R = length(Fhat);
err = zeros(1,R);
for r = 1:R
    % first, normalize
    Fhat{r} = Fhat{r}*diag(1./sqrt(sum(abs(Fhat{r}).^2,1)));
    RA = sqrt(sum(abs(Fref{r}).^2,1));
    Fref{r} = Fref{r}*diag(1./RA);
    d = size(Fhat{r},2);
    % second, fix permutation
    fromassign = 1:d;
    toassign = 1:d;
    order = zeros(1,d);
    for n = 1:d
        A = abs(Fhat{r}(:,toassign)'*Fref{r}(:,fromassign));
        [m,w] = max(A(:)); %#ok<ASGLU>
        [mx,my] = ind2sub([d-n+1,d-n+1],w);
        order(fromassign(my)) = toassign(mx);
        toassign = toassign([1:mx-1,mx+1:end]);
        fromassign = fromassign([1:my-1,my+1:end]);
    end
    
    % third, fix phase (+scaling, again):
    alpha = zeros(1,d);
    for n = 1:d
        alpha(n) = Fhat{r}(:,order(n))'*Fref{r}(:,n);
    end
    Fhat{r} = Fhat{r}(:,order)*diag(alpha);
    
    
    % finally, output error
    err(r) = norm(Fhat{r}(:)-Fref{r}(:))^2/norm(Fref{r}(:))^2;
    
    Fhat{r} = Fhat{r}*diag(RA);
end
if nc
    Fhat = Fhat{1};
end