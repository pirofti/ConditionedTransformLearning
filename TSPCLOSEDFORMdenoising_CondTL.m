function [IMU,paramsout]= TSPCLOSEDFORMdenoising_CondTL(I1,I7,paramsin)

% Implementation adapted on the transform learning denoising framework from
% S. Ravishankar and Y. Bresler, ``\ell_0 Sparsifying transform learning with
% efficient optimal updates and convergence guarantees'',
% IEEE Transactions on Signal Processing, vol. 63, no. 9, pp. 2389-2404, May 2015.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Inputs: 1) I1 : Noisy image
%        2) I7 : Noiseless reference
%        3) paramsin: Structure that contains the parameters of the denoising algorithm. The various fields are as follows -
%                   - sig: Standard deviation of the i.i.d. Gaussian noise
%                   - iterx: Number of iterations of the two-step denoising algorithm (e.g., iterx= 11)
%                   - n: Patch size, i.e., Total number of pixels in a square patch (e.g., n= 121, 64)
%                   - N: Number of training signals used in the transform learning step of the algorithm (e.g., N=500*64)
%                   - C: Parameter that sets the threshold that determines sparsity levels in the variable sparsity update step (e.g., C=1.04 when n=121; C = 1.08 when n=64)
%                   - s: Initial sparsity level for patches (e.g., s=round((0.1)*n))
%                   - tau: Sets the weight \tau in the algorithm (e.g., tau=0.01/sig)
%                   - maxsparsity: Maximum sparsity level allowed in the variable sparsity update step of the algorithm (e.g., maxsparsity = round(6*s))
%                   - M: Number of iterations within transform learning step (e.g., M = 12)
%                   - method: If set to 0, transform learning is done employing a log-determinant+Frobenius norm regularizer.
%                   	If set to 1,  an orthonormal transform learning procedure is adopted
%                   	If set to 2--7, a RIP variant is adopted
%                   - lambda0: Determines the weight on the log-determinant+Frobenius norm regularizer. To be used in the case when  = 0. (e.g., lambda0=0.031)
%                   - W: Initial transform in the algorithm  (e.g., W=kron(dctmtx(sqrt(n)),dctmtx(sqrt(n))))
%                   - r: Patch Overlap Stride (e.g., r=1)
%

%Outputs:  1) IMU: Denoised image
%          2) paramsout - Structure containing outputs other than the denoised image.
%                 - PSNR : PSNR of denoised image.
%                 - transform : learnt transform

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initializing algorithm parameters
[aa,bb]=size(I7);
sig = paramsin.sig;
iterx = paramsin.iterx;
n = paramsin.n;
N = paramsin.N;
C1 = paramsin.C;
la = paramsin.tau;
T0=  paramsin.s;
maxsp = paramsin.maxsparsity;
numiterr = paramsin.M;
% if(paramsin.method==1)
    lambda0=paramsin.lambda0;
% end
W =  paramsin.W;
r = paramsin.r;

Wbresles = W;
Wlast = W;

threshold=C1*sig*(sqrt(n)); %\ell_2 error threshold (maximum allowed norm of the difference between a noisy patch and its denoised version) per patch

%Initial steps

%Extract image patches
[TE,idx] = my_im2col(I1,[sqrt(n),sqrt(n)],r); br=mean(TE);
TE=TE - (ones(n,1)*br); %subtract means of patches
[rows,cols] = ind2sub(size(I1)-sqrt(n)+1,idx);

N4=size(TE,2); %Total number of overlapping image patches

%Check if input training size exceeds total number of image patches
if(N4>N)
    N3=N;
else
    N3=N4;
end


de=randperm(N4);

YH=TE(:,de(1:N3));  %Use a random subset of patches in the transform learning step
STY =(ones(1,N3))*(T0);  %Vector of initial sparsity levels


%Begin iterations of the two-step denoising algorithm
for ppp=1:iterx
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    l2=(lambda0)*((norm(YH,'fro'))^2); l3=l2;
    [W]= TLclosedformmethod(Wbresles,YH,numiterr,l2,l3,STY);  %Transform Learning with log-determinant+Frobenius norm regularizer
    condW = cond(W);
    froW = norm(W,'fro');
    Wbresles = W;
    %Transform Learning Step
    if(strcmp(paramsin.method, 'TLORTHO'))
        [W]= TLORTHO(Wlast,YH,numiterr,STY);  %Orthonormal Transform Learning
    elseif(strcmp(paramsin.method, 'TLRIP_hard_way_simplified'))
    	[W,~,~]= TLRIP_hard_way_simplified(Wlast,YH,numiterr,STY,condW,froW);
    end
    Wlast = W;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    de=randperm(N4);
    
    if(ppp==iterx)
        X1=W*TE;   %In the last iteration of the two-step denoising algorithm, apply the transform to all patches
        kT=zeros(1,size(TE,2));
    else
        YH=TE(:,de(1:N3));  %In all but the last iteration of the two-step denoising algorithm, select a random subset of patches (for use in training in the next iteration)
        X1=W*(YH);
        kT=zeros(1,size(YH,2));
    end
    
    [~,ind]=sort(abs(X1),'descend');
    er=n*(0:(size(X1,2)-1));ind=ind + (er'*ones(1,n))';
    G=(pinv([(sqrt(la)*eye(n));W])); Ga=G(:,1:n);Gb=G(:,n+1:2*n);
    
    %Variable Sparsity Update Step
    if(ppp==iterx)
        Gz=Ga*((sqrt(la))*TE);
        q=Gz;   %In the last iteration of the two-step denoising algorithm, q stores the denoised patches.
        ZS2=sqrt(sum((Gz-TE).^2));
        kT=kT+(ZS2<=threshold);  %Checks if error threshold is satisfied at zero sparsity for any of the patches
        STY=zeros(1,size(TE,2));X=zeros(n,size(TE,2)); %STY is a vector of sparsity levels and X is the corresponding sparse code matrix
    else
        Gz=Ga*((sqrt(la))*YH);
        ZS2=sqrt(sum((Gz-YH).^2));
        kT=kT+(ZS2<=threshold);  %Checks if error threshold is satisfied at zero sparsity for any of the training patches
        STY=zeros(1,size(YH,2));X=zeros(n,size(YH,2)); %STY is a vector of sparsity levels and X is the corresponding sparse code matrix
    end
    
    %Incrementing sparsity by 1 at a time, until error threshold is satisfied for all patches.
    for k=1:maxsp
        indi=find(kT==0); %Find indices of patches for which the error threshold has not yet been satisfied
        if(isempty(indi))
            break;
        end
        
        X(ind(k,indi))=X1(ind(k,indi));  %Update sparse codes to the current sparsity level in the loop.
        if(ppp==iterx)
            q(:,indi)= Gz(:,indi) + Gb*(X(:,indi));  %Update denoised patches in the last iteration of the two-step denoising algorithm
            ZS2=sqrt(sum((q(:,indi) - TE(:,indi)).^2)); kT(indi)=kT(indi)+(ZS2<=threshold);  %Check if error threshold is satisfied at sparsity k for any patches
        else
            ZS2=sqrt(sum((Gz(:,indi) + Gb*(X(:,indi)) - YH(:,indi)).^2)); kT(indi)=kT(indi)+(ZS2<=threshold); %Check if error threshold is satisfied at sparsity k for any training patches
        end
        STY(indi)=k;  %Update the sparsity levels of patches
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Averaging together the denoised patches at their respective locations in the 2D image
IMout=zeros(aa,bb);Weight=zeros(aa,bb);
bbb=sqrt(n);
for jj = 1:10000:size(TE,2)
    jumpSize = min(jj+10000-1,size(TE,2));
    ZZ= q(:,jj:jumpSize) + (ones(size(TE,1),1) * br(jj:jumpSize));
    inx=(ZZ<0);ing= ZZ>255; ZZ(inx)=0;ZZ(ing)=255;
    for ii  = jj:jumpSize
        col = cols(ii); row = rows(ii);
        block =reshape(ZZ(:,ii-jj+1),[bbb,bbb]);
        IMout(row:row+bbb-1,col:col+bbb-1)=IMout(row:row+bbb-1,col:col+bbb-1)+block;
        Weight(row:row+bbb-1,col:col+bbb-1)=Weight(row:row+bbb-1,col:col+bbb-1)+ones(bbb);
    end
end
IMU=(IMout)./(Weight);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ing= IMU<0; ing2= IMU>255;
IMU(ing)=0;IMU(ing2)=255;  %Limit denoised image pixel intensities to the range [0, 255].


%Output the learnt transform and the PSNR of the denoised image.
paramsout.transform=W;
paramsout.PSNR=20*log10((sqrt(aa*bb))*255/(norm(double(IMU)-double(I7),'fro')));
