function [correct_emission_predictions, state_predictions] = predict_onestep(model,data)
% correct_emission_predictions is a  num_subparts x T array, where element
% (i,t) is predicted probability that answer to subpart i at time t+1 is correct

import fit.*

num_subparts = size(data.data,1);
num_resources = length(model.learns);

trans_softcounts = zeros(2,2,num_resources);
emission_softcounts = zeros(2,2,num_subparts);
init_softcounts = zeros(2,1);

[~, forward_messages] = E_step(data,model,...
                               trans_softcounts,emission_softcounts,init_softcounts);
state_predictions = predict_onestep_states(data,model,forward_messages);

correct_emission_predictions = bsxfun(@times,model.guesses(:),state_predictions(1,:)) ...
                                + bsxfun(@times,1-model.slips(:),state_predictions(2,:));

end
