function compare_edxhmm_to_bnt
% 
addpath(genpath('./bnt/'))
%load edxhmm_vs_bnt;
m=generate.random_model;
m2=generate.random_model;
d=generate.synthetic_data(m,ones(1,50)*4);

[edx,edxll]=fit.EM_fit(m2,d,0);

disp('Training data');
d
disp('training initial model')
m2

users=size(d.starts,1);
data=zeros(users,max(d.lengths)+1);
data(:,1) = 1:users';
for s=1:users
	data(s,2:5) = d.data(d.starts(s):d.starts(s)+d.lengths(s)-1)-1;
end

if (ischar(dataset))
	data1 = load(dataset);
else
	data1 = dataset;
end
	
N = 8;
dag = zeros(N,N);
dag(1,2) = 1;
dag(2,3) = 1;
dag(3,4) = 1;
dag(1,5) = 1;
dag(2,6) = 1;
dag(3,7) = 1;
dag(4,8) = 1;

observed = [5:8];
discrete_nodes = 1:N;
node_sizes = 2*ones(1,N);
equiv_class = [1 2 2 2 3 3 3 3];
bnet = mk_bnet(dag, node_sizes, 'discrete', discrete_nodes, 'observed', observed, 'equiv_class', equiv_class);
guess = m2.guesses;
slip = m2.slips;
learn = m2.learn;
forget = m2.forget;
prior = m2.prior;
bnet.CPD{1} = tabular_CPD(bnet, find(equiv_class == 1,1), 'CPT', [1-prior prior]);
bnet.CPD{2} = tabular_CPD(bnet, find(equiv_class == 2,1), 'CPT', [1-learn forget learn 1-forget]);
bnet.CPD{3} = tabular_CPD(bnet, find(equiv_class == 3,1), 'CPT', [1-guess slip guess 1-slip]);

cases = cell(size(data,1),N);
c = 1;
for d=1:size(data,1)
  cases(c,[5:8]) = num2cell(data(d,2:5)+1);
  c = c + 1;
end
thres = 1e-3;
max_iter = 100;
engine = jtree_inf_engine(bnet);
[bnet2, bntll] = learn_params_em(engine,cases',max_iter,thres);

iprior = CPD_to_CPT(bnet2.CPD{1});
iprior = iprior(2);

iplearn1 = CPD_to_CPT(bnet2.CPD{2});
ipforget = iplearn1(2);
iplearn1 = iplearn1(3);

ipguess = CPD_to_CPT(bnet2.CPD{3});
ipslip = ipguess(2);
ipguess = ipguess(3);

model.prior=iprior;
model.learn=iplearn1;
model.forget=ipforget;
model.guesses=ipguess;
model.slips=ipslip;
bnt=model;
%fprintf('prior\tlearn\tguess\tslip\n');
%fprintf('%.4f\t%.4f\t%.4f\t%.4f\n',iprior,iplearn1,ipguess,ipslip);

disp('edX HMM fit model')
edx
disp('BNT EM fit model')
bnt
disp('truth model')
m
disp('difference between truth and:')
fprintf('\tedxhmm\tBNT\n');
fprintf('learn\t%.4f\t%.4f\n',m.learn-edx.learn,m.learn-bnt.learn);
fprintf('forget\t%.4f\t%.4f\n',m.forget-edx.forget,m.forget-bnt.forget);
fprintf('guess\t%.4f\t%.4f\n',m.guesses-edx.guesses,m.guesses-bnt.guesses);
fprintf('slip\t%.4f\t%.4f\n',m.slips-edx.slips,m.slips-bnt.slips);
fprintf('LL\t%.2f\t%.2f\n',edxll(end),bntll(end));
