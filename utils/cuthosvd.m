function [Sc,Uc] = cuthosvd(S,U,d)

% CUTHOSVD   Truncate HOSVD to d elements.
%
% Syntax:
%   [S_c, U_c] = CUTHOSVD(S,U,d)
%
% Input:
%   S: HOSVD core tensor 
%   U: HOSVD singular vector matrices (stored in a cell array)
%   d: number of elements to truncate to
%
% Output:
%   S_c: core tensor truncated to d elements (see HELP CUTTENSOR).
%   U_c: singular vector matrices truncated to d columns.
%
% Example:
%   For a tensor X of size [I1,I2,I3]:
%     [S,U] = hosvd(X) results in S of size [I1,I2,I3] and U of size {1,3},
%     where size(U{1}) = [I1,I1], size(U{2}) = [I2,I2], size(U{3}) = [I3,I3].
%   Now, [Sc,Uc] = cuthosvd(S,U,d) results in Sc of 
%   size [min(d,I1),min(d,I2),min(d,I3)] and for U in size(U{1}) = [I1,min(d,I1)],
%   size(U{2}) = [I2,min(d,I2)], and size(U{3}) = [I3,min(d,I3)]
%   
% Author:
%    Florian Roemer, Communications Resarch Lab, TU Ilmenau
% Date:
%    Dec 2007

Sc = cuttensor(S,d);
Uc = cell(size(U));
for n = 1:length(U)
    Uc{n} = U{n}(:,1:min(d,size(U{n},2)));
end