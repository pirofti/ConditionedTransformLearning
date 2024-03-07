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

function [k1, k2, l_star] = kappa_exhaustive_search(r_over_d, d, r, kappa)
n = length(r_over_d);

if r_over_d(1)/r_over_d(end) <= kappa
    k1 = 0; k2 = 0; l_star = 0;
    return;
end

for k2 = 1:n-1
    for k1 = 1:n-k2
        l_star = get_l_star(r_over_d, d, r, k1, k2, kappa);
        u_star = kappa*l_star;
        
        y = 0;
        
        if r_over_d(end-k1) >= l_star
            y = 1;
        else
            y = 0;
        end
        
        if r_over_d(k2+1) > u_star
           y = 0;
        end 
        
        if y
            return;
        end
    end
end

k1 = n;
k2 = 0;
l_star = get_l_star(r_over_d, d, r, k1, k2, kappa);

stop = 1;
