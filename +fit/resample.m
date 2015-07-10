function [model, log_likelihoods] = resample(model,data,numiter,temps,priors)

import fit.*

num_subparts = size(data.data,1);
num_resources = length(model.learns);

if ~exist('priors','var')
    priors = struct;
end
if ~isfield(priors,'As')
    priors.As = ones(2,2,num_resources);
end
if ~isfield(priors,'emissions')
    priors.emissions = ones(2,2,num_subparts);
end
if ~isfield(priors,'inits')
    priors.inits = ones(2,1);
end
if ~exist('numiter','var'), numiter=50; end
if ~exist('temps','var'), temps=max(ones(1,numiter),linspace(20,-20,numiter)); end

trans_counts = zeros(2,2,num_resources);
emission_counts = zeros(2,2,num_subparts);
init_counts = zeros(2,1);
log_likelihoods = zeros(numiter,1);

for i=1:numiter
    log_likelihoods(i) = sample_states(data, model,...
                                trans_counts, emission_counts, init_counts,...
                                temps(i));
    model = resample_params(priors,trans_counts,emission_counts,init_counts,temps(i));
end

end
