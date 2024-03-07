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

function [solution, V] = get_spectrum_from_data(U, V, X, Y, kappa, the_sum, l_min)
n = size(U, 1);

d = zeros(n, 1);
for i = 1:n
    d(i) = norm(Y'*V(:,i));
end

r = zeros(n, 1);
Q = Y*X';
for i = 1:n
    r(i) = V(:, i)'*Q*U(:, i) / norm(Y'*V(:, i));
    if r(i) < 0.0
      r(i) = -r(i);
      V(:,i) = -1 * V(:, i);
    end
end

r_over_d = r./d;
[r_over_d_sorted, r_over_d_indices] = sort(r_over_d, 'descend');

[k1, k2, l_star] = kappa_exhaustive_search(r_over_d_sorted, d(r_over_d_indices), r(r_over_d_indices), kappa);
if l_star~=0
    solution = r_over_d_sorted;
    solution(1:k2) = kappa*l_star;
    solution(end:-1:end-k1+1) = l_star;
else
    solution = r_over_d_sorted;
end
new_indices(r_over_d_indices) = 1:length(solution);
solution = solution(new_indices);

stop = 1;
