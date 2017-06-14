function [ loss, gradient ] = COINS_AllPairExpansion( wb, dataU, epsilon )
%
% INPUT:
% wb    (nC,2*dim+2) weight vector
% xU    (dim,n)    unlabeled inputs
% epsilon   hyperparameter in balcan's epsilon expansion theory (Balcan et. al, 2004)
%
% OUTPUT:
% loss          loss
% gradient      (nC, 2*(dim+1))     gradient
%
% VARIABLES:
% num_class     number of label
%
    [num_unlabel_data, dim] = size(dataU);
    [num_class, dim2] = size(wb);
    lambda2 = 10/(num_class*(num_class-1));

    loss = 0;
    gradient = zeros(num_class, dim2);

    dataUb = [dataU, ones(num_unlabel_data,1)];
    wb1 = wb(:, 1:dim+1);
    wb2 = wb(:, dim+2:end);
    value1 = wb1*dataUb';
    value2 = wb2*dataUb';

    for i = 1:num_class-1
        for j = i+1:num_class
            [loss1, grad1] = COINS_balcanExpansion(wb, i, j, value1, value2, dataU, epsilon);
            [loss2, grad2] = COINS_balcanExpansion(wb, j, i, value1, value2, dataU, epsilon);
            loss = loss + loss1 + loss2;
            gradient = gradient + grad1 + grad2;
        end
    end
    loss = loss*lambda2;
    gradient = gradient*lambda2;
end

