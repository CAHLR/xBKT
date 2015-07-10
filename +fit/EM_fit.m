function [model, log_likelihoods] = EM_fit(model,data,tol,maxiter)
import fit.*

if ~exist('tol','var'), tol = 1e-3; end
if ~exist('maxiter','var'), maxiter=100; end

util.check_data(data)

num_subparts = size(data.data,1);
num_resources = length(model.learns);

trans_softcounts = zeros(2,2,num_resources);
emission_softcounts = zeros(2,2,num_subparts);
init_softcounts = zeros(2,1);
log_likelihoods = zeros(maxiter,1);

for i=1:maxiter
    log_likelihoods(i) = E_step(data, model,...
                                trans_softcounts, emission_softcounts, init_softcounts);
    if (i > 1 && abs(log_likelihoods(i) - log_likelihoods(i-1)) < tol)
        break
    end
    model = M_step(trans_softcounts,emission_softcounts,init_softcounts);
end

log_likelihoods = log_likelihoods(1:i);

end
