function [ top_n_num, idx ] = topN( arr, N, mode )
%
% mode selects the direction of the sort
%      'ascend' results in ascending order
%      'descend' results in descending order

    if nargin<2
        error('Not enough input parameters, please check again.');
    end
    if nargin<3
        mode = 'ascend';
    end
    
    if strcmp(mode, 'ascend')
        cmp_func = @min;
        sort_mode = 'descend';
    else
        if strcmp(mode, 'descend')
            cmp_func = @max;
            sort_mode = 'ascend';
        else
            error('wrong mode.');
        end
    end
    if N>length(arr)
        error('N is bigger than a'' length.');
    end
    [ top_n_num, idx ] = sort(arr, mode);
    top_n_num = top_n_num(1:N);
    idx = idx(1:N);
end

