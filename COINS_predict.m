function [ confidence, pre_labels, confidenceA, confidenceB ] = COINS_predict( wb, dataX )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [num_data, dim] = size(dataX);
    [num_class,~] = size(wb);
    wbA = wb(:,1:dim+1);
    wbB = wb(:,dim+2:end);
    confidenceA = COINS_predict_one_view( wbA, dataX );
    confidenceB = COINS_predict_one_view( wbB, dataX );
    confidence = (confidenceA + confidenceB)/2;
    pre_labels = zeros(num_class-1, num_data);
    pre_labels(confidence>=0.5)=1;
    pre_labels(confidence<0.5)=-1;
end

