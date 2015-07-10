function datastruct = yanbox_convert_data(data,lengths)

resources = int16(data(:,1)');
lengths = int32(lengths(:));
starts = int32([1;cumsum(lengths(1:end-1))+1]);
data = int8(data(:,2:end)');

datastruct = struct;
datastruct.data = data;
datastruct.resources = resources;
datastruct.starts = starts;
datastruct.lengths = lengths;


end
