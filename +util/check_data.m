function check_data(data)
    if ~(size(data.data,2) == size(data.resources,2))
        error('data and resource sizes must match')
    end
    if ~all(data.starts + data.lengths - 1 <= size(data.data,2))
        error('data lengths don''t match its shape')
    end
end
