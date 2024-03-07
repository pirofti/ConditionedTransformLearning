% Copyright (c) 2023-2024 Paul Irofti <paul@irofti.net>
% Copyright (c) 2023-2024 Cristian Rusu <cristian.rusu@fmi.unibuc.ro>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

function [W,XB,error,error2] = ConditionedTransformLearning(W,Y,numiter, STY, kappa, targetWnorm)

% Implementation adapted on the transform learning framework from the series of
% papers by S. Ravishankar and Y. Bresler

%We employ alternating minimization here to solve a transform learning problem that involves a constraint on the adaptive transform domain sparsity of each training signal, 
%and a constraint that the transform is unitary or orthonormal.
%The algorithm iterates over a sparse coding step and a transform update step, both of which involve efficient update procedures.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Inputs: 1) W : Initial Transform
%        2) Y : Training Matrix with signals as columns
%        3) numiter:  Number of iterations of alternating minimization
%        4) STY: Vector containing maximum allowed sparsity levels for each training signal.

%Outputs:  1) W: Learnt Transform
%          2) XB: Learnt Sparse Code


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initial steps
[K,n]=size(W);XB=zeros(K,size(Y,2));

ix=find(STY>0); q=Y(:,ix); STY=STY(:,ix); N=size(q,2);
ez=K*(0:(N-1));STY=STY + ez;
Y = Y(:, ix);

error = [];
error2 = [];
kappas = [];

%Algorithm iterations in a FOR Loop
for i=1:numiter

    %Sparse Coding Step
    X1=W*q;
    [s]=sort(abs(X1),'descend');
    X = X1.*(bsxfun(@ge,abs(X1),s(STY)));
    
    %Transform Update Step
    if isempty(kappas)
        kappas = linspace(numiter*kappa, kappa, numiter);
        
        if kappa <= 2
            W = Y*X';
            [U, S, V] = svd(W);
        else
            W = (Y'\X')';
            [U, S, V] = svd(W);
        end
        
        the_sum = sum(diag(S));
        l_min = min(diag(S));
        
        [d_cvx, V] = get_spectrum_from_data(U, V, X, Y, kappa, the_sum, l_min);
    end
    
    %%% get the U
    [Q, ~, T] = svd(diag(d_cvx)*V'*Y*X');
    U = (Q*T')';
    
    %%% get the V
    [Uu, Ss, Vv] = svd((U'*X)*Y');
    V = (Uu*Vv')';
    
    %%% get the spectrum
    [d_cvx, V] = get_spectrum_from_data(U, V, X, Y, kappa, the_sum, l_min);
    
    W = U*diag(d_cvx)*V';
    W = W/norm(W, 'fro')*targetWnorm;
    
    error = [error norm(X - W*Y, 'fro')];
    error2 = [error2 norm(X - W*Y, 'fro')/norm(W*Y, 'fro')];
    if (i == 84)
        stop = 1;
    end
end
XB(:,ix)=X;
