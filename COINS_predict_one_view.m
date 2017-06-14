function [ confidence, pre_labels ] = COINS_predict_one_view( wb, dataX )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [num_data, ~] = size(dataX);
    [num_class,~] = size(wb);
    dataX = [dataX,ones(num_data,1)];

    output = wb*dataX';
    diff = output(1:end-1,:) - repmat(output(end,:),num_class-1,1);
    confidence = 1./(1+exp(-diff));
    pre_labels = zeros(num_class-1, num_data);
    pre_labels(confidence>=0.5)=1;
    pre_labels(confidence<0.5)=-1;
end

