% Copyright (c) 2023-2024 Paul Irofti <paul@irofti.net>
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

close all
clear
clc
noisy_images = {'Cameraman' 'Couple' 'Hill' 'Lena' 'Man' 'MRI' 'Barbara'};

logfile = fopen('condtl_deniosing.log','w');

for iimg = 1:length(noisy_images)
iimage = noisy_images{iimg}

for sigma=[5 10 15 20 100]

sigma_path = ['./img/' iimage '/sigma' num2str(sigma)]

addpath(sigma_path)

load I1  %noisy image
load I7  %noiseless reference

% methods to try denoising with
methods = {'ConditionedTransformLearning', 'Bresler', 'TLORTHO'};
n_methods = length(methods);

sig=sigma;  %standard deviation of Gaussian noise
n=121;  %patch size (number of pixels in patch) to use (e.g., n = 121, 64).

s=round((0.1)*n);
if(sig>=100)
    iter=5;
else
    iter=20;
end

%initialize parameters
paramsin.sig = sig;
paramsin.iterx = iter;
paramsin.n = n;
paramsin.N = 500;
if(n==64)
    C=1.08;
end
if(n==121)
    C=1.04;
end
C=1.15;
paramsin.C = C;
paramsin.tau = 0.01/sig;
paramsin.s = s;
paramsin.M = 12;
paramsin.maxsparsity = round(6*s);
paramsin.method = 0;
paramsin.lambda0 = 0.0031;
paramsin.W = kron(dctmtx(sqrt(n)),dctmtx(sqrt(n)));
paramsin.r = 1;

images = cell(n_methods);
Ws = cell(n_methods);
psnrs = cell(n_methods);
ssims = cell(n_methods);
%Denoising algorithm
for method=1:n_methods
	paramsin.method = methods{method};
	%IMU is the output denoised image
	[IMU,paramsout]= TSPCLOSEDFORMdenoising_CondTL(I1,I7,paramsin);
	images{method} = IMU;
	Ws{method} = paramsout.transform;
	psnrs{method} = paramsout.PSNR;
        [ssims{method}, ~] = ssim(IMU, I7);
        fprintf(logfile, '[%s,%d,%d] %s psnr=%f ssim=%f\n', ...
		iimage, sigma, method, methods{method}, ...
		psnrs{method}, ssims{method});
	fflush(logfile);
        fprintf('[%s,%d,%d] %s psnr=%f ssim=%f\n', ...
		iimage, sigma, method, methods{method}, ...
		psnrs{method}, ssims{method});
end

rmpath(sigma_path)
end
end

fclose(logfile)
