function mu = softmax(eta)
    % Softmax function
    % mu(i,c) = exp(eta(i,c))/sum_c' exp(eta(i,c'))

    % This file is from matlabtools.googlecode.com
    c = 1;

    tmp = exp(c*eta);
    denom = sum(tmp, 1);
    mu = bsxfun(@rdivide, tmp, denom);

end
% function softmaxA = softmax(A,dim)
% 
% % softmax computes the softmax of the vectors of the matrix A, taken
% % along dimension dim.
% 
% if nargin < 2, error('The dimension along which to do the softmax must be provided.'); end
% 
% s = ones(1, ndims(A));
% s(dim) = size(A, dim);
% 
% % First get the maximum of A.
% maxA = max(A, [], dim);
% expA = exp(A-repmat(maxA, s));
% softmaxA = expA ./ repmat(sum(expA,dim), s);
% % function mu = softmax(eta)
% %     % Softmax function
% %     % mu(i,c) = exp(eta(i,c))/sum_c' exp(eta(i,c'))
% % 
% %     % This file is from matlabtools.googlecode.com
% %     c = 3;
% % 
% %     tmp = exp(c*eta);
% %     denom = sum(tmp, 2);
% %     mu = bsxfun(@rdivide, tmp, denom);
% % 
% % end