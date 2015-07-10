function [pred,d3,mt] = compare_edxhmm_to_bnt2
% 
addpath(genpath('./bnt/'))
%load edxhmm_vs_bnt;
m = generate.random_model(1);
d = generate.synthetic_data(m,ones(1,400)*8);
d2 = generate.synthetic_data(m,ones(1,100)*8);

bestll = -inf;
for i=1:25
    [m2,ll] = fit.EM_fit(generate.random_model(1),d,0);
    if (ll(end) > bestll)
	mt = m2;
	bestll = ll(end);
    end
end
	
pred = struct;

% predicting all 8 actions of each 100 users in the test data
pred.new_all = fit.predict_onestep(mt,d2);
pred.bnt_all = bnt_predict(mt,d2);

% re-organize the test data to only include the first half of responses of each user
d3 = d2;
d3.lengths=round(d2.lengths/2);
d3.starts = d2.starts-[0:length(d2.starts)-1]'*(d.lengths(1)-d3.lengths(1));
d3.data = [];
% this is lame and there's got to be a nicer way
for i=1:length(d2.starts)
    d3.data = [d3.data d2.data(d2.starts(i):d2.starts(i)+d3.lengths(i)-1)];
end

% re-organize prediction data to only include the first half of responses for each user
pred.new_half = [];
pred.bnt_half = [];
for i=1:length(d2.starts)
    pred.new_half = [pred.new_half pred.new_all(d2.starts(i):d2.starts(i)+d3.lengths(i)-1)];
    pred.bnt_half = [pred.bnt_half pred.bnt_all(d2.starts(i):d2.starts(i)+d3.lengths(i)-1)];
end

% predict the first half without being give the second half
pred.new_half2 = fit.predict_onestep(mt,d3);
pred.bnt_half2 = bnt_predict(mt,d3);

fprintf('ALL\tNEW\tBNT\n');
fprintf('RMSE\t%.4f\t%.4f\n',sqrt(mean((pred.new_all-(d2.data-1)).^2)),sqrt(mean((pred.bnt_all-(d2.data-1)).^2)));
fprintf('CORR\t%.4f\t%.4f\n',corr(d2.data',pred.new_all'),corr(d2.data',pred.bnt_all'));

fprintf('\nHALF\tNEW\tBNT\n');
fprintf('RMSE\t%.4f\t%.4f\n',sqrt(mean((pred.new_half-(d3.data-1)).^2)),sqrt(mean((pred.bnt_half-(d3.data-1)).^2)));
fprintf('CORR\t%.4f\t%.4f\n',corr(d3.data',pred.bnt_half'),corr(d3.data',pred.bnt_half'));

fprintf('\nHALF2\tNEW\tBNT\n');
fprintf('RMSE\t%.4f\t%.4f\n',sqrt(mean((pred.new_half2-(d3.data-1)).^2)),sqrt(mean((pred.bnt_half2-(d3.data-1)).^2)));
fprintf('CORR\t%.4f\t%.4f\n',corr(d3.data',pred.bnt_half2'),corr(d3.data',pred.bnt_half2'));

function predictions = bnt_predict(mt,d0)

max_length = max(d0.lengths);
users=size(d0.starts,1);
data2=zeros(users,max(d0.lengths)+1);
data2(:,1) = 1:users';
for s=1:users
	data2(s,2:max_length+1) = d0.data(d0.starts(s):d0.starts(s)+d0.lengths(s)-1)-1;
end

N=max_length*2;
hidden_nodes=1:N/2;
dag=zeros(N,N);
for k=1:length(hidden_nodes)-1
 dag(hidden_nodes(k),hidden_nodes(k)+1) = 1;
end

for k=hidden_nodes
 dag(k,k+N/2) = 1;
end

observed_nodes = (N/2+1):N;
discrete_nodes = 1:N;
node_sizes = 2*ones(1,N);
equiv_class= [1 2*ones(1,length(hidden_nodes)-1) 3*ones(1,length(observed_nodes))];
bnet = mk_bnet(dag, node_sizes, 'discrete', discrete_nodes, 'observed', observed_nodes, 'equiv_class', equiv_class);
guess = mt.guesses;
slip = mt.slips;
learn = mt.learn;
forget = mt.forget;
prior = mt.prior;
bnet.CPD{1} = tabular_CPD(bnet, find(equiv_class == 1,1), 'CPT', [1-prior prior]);
bnet.CPD{2} = tabular_CPD(bnet, find(equiv_class == 2,1), 'CPT', [1-learn forget learn 1-forget]);
bnet.CPD{3} = tabular_CPD(bnet, find(equiv_class == 3,1), 'CPT', [1-guess slip guess 1-slip]);
engine2 = jtree_inf_engine(bnet);
% create test evidence array
cases2 = cell(size(data2,1),N);
for c=1:size(data2,1)
 for r=1:size(data2,2)-1
   cases2(c,observed_nodes(r)) = num2cell(data2(c,r+1)+1);
 end
end

pindx = 1;
predictions = zeros(size(data2,1)*(size(data2,2)-1),3);
for c=1:size(data2,1)
 scase=cell(1,N);
 for r=1:size(data2,2)-1
   for ev=1:r-1 
    scase(observed_nodes(ev)) = cases2(c,observed_nodes(ev));
   end
  
   [engine3,ll] = enter_evidence(engine2,scase);
   m = marginal_nodes(engine3,observed_nodes(r));
   m = m.T(2);
   predictions(pindx,1) = data2(c,1);
   predictions(pindx,2) = m;
   predictions(pindx,3) = data2(c,r+1);
   pindx = pindx+1;
  end
end

%fprintf('\nverify that data seq aligns with prediction seq\n');
%corr(d0.data',predictions(:,3))
predictions = predictions(:,2)';
