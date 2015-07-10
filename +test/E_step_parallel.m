import fit.*

num_resources = 2;
num_subparts = 1;
model = generate.random_model(num_resources,num_subparts);
disp('generating data...')
data = generate.synthetic_data(model,100*ones(1,1000000));
disp('...done!')

trans_softcounts = zeros(2,2,num_resources);
emission_softcounts = zeros(2,2,num_subparts);
init_softcounts = zeros(2,1);

disp('running serial...')
tic; [l_serial, a_serial, g_serial] = fit.E_step_serial(data,model,trans_softcounts,emission_softcounts,init_softcounts); toc;
disp('...done!')
disp('running parallel...')
tic; [l, a, g] = fit.E_step(data,model,trans_softcounts,emission_softcounts,init_softcounts); toc;
disp('...done!')

abs(l - l_serial) < 1

all(a(:) == a_serial(:))

all(g(:) == g_serial(:))

