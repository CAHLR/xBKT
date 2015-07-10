function tot = predict_and_compare(model,data)
% returns the proportion of incorrect predictions (0 is perfect, 1 is all bad)

import fit.*

predicted_correct_ans_probs = predict_onestep(model,data);

tot = 0;
for startlen=[data.starts;data.lengths]
    start = startlen(1);
    len = startlen(2);

    d = data.data(:,start:start+len-1);
    p = predicted_correct_ans_probs(:,start:start+len-1);
    p = p(d ~= 0);
    d = d(d ~= 0);
    tot = tot + sum(abs((d == 2) - p),2);
end

% TODO handle missing data
tot = tot ./ (size(data.data,2) - length(data.lengths));

end
