function datastruct = synthetic_data(model, lengths, resources)

import generate.*

num_resources = size(model.learns,3);
bigT = sum(lengths);

if ~exist('resources','var'), resources = randi(num_resources,1,bigT); end
if ~isfield(model,'As')
    model.As = [1-model.learns, model.forgets; model.learns, 1-model.forgets];
end

if ~isfield(model,'pi_0'), model.pi_0 = [1-model.prior; model.prior]; end

resources = int16(resources);
lengths = int32(lengths(:));
starts = int32([1;cumsum(lengths(1:end-1))+1]);

[stateseqs, data] = synthetic_data_helper(model,starts,lengths,resources);
data = data + 1; % 1 and 2 instead of 0 and 1kf

data(:,resources ~= 1) = 0; % no data emitted unless resource == 1

datastruct = struct;
datastruct.stateseqs = stateseqs;
datastruct.data = data;
datastruct.starts = starts;
datastruct.lengths = lengths;
datastruct.resources = resources;

end
