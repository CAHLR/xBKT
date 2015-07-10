num_subparts = 4;
num_resources = 2;

model = generate.random_model(num_resources,num_subparts);

%% M step

trans_softcounts = model.As*100;

% in matlab, smallest step is incrementing the first index
% 2x2xnum_subparts
% first index is state, second index is emission, third index is subpart idx
emission_softcounts = zeros(2,2,num_subparts);
emission_softcounts(1,2,:) = model.guesses; % in state 1 (don't know) but get emission 2 (correct)
emission_softcounts(1,1,:) = 1-model.guesses;
emission_softcounts(2,1,:) = model.slips;
emission_softcounts(2,2,:) = 1-model.slips;
emission_sofcounts = emission_softcounts * 100;

init_softcounts = [99; 1];

fitmodel = fit.M_step(trans_softcounts,emission_softcounts,init_softcounts);

%% E step


