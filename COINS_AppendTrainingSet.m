function [ apd_train_idx_B, apd_rel_label_B, apd_irr_label_B, apd_penalty_B ] = COINS_AppendTrainingSet( wbA, dataX, dataY, unlabel_train_idx, step_size )
% INPUT:
% wb        (nC, 2*dim+2)   weight and bias of two views, [wbA, wbB]
% dataX     (nLT, dim)      training data
% dataY     (nC, nLT)       labels of training data

    belief_threshold = 0.8;

    dataU = [dataX(unlabel_train_idx,:),ones(length(unlabel_train_idx),1)];

    [rankA, thresholdA] = getData(wbA, dataU, dataY, unlabel_train_idx);

    [apd_train_idx_B, apd_rel_label_B, apd_irr_label_B, apd_penalty_B] = getAppendData( rankA, thresholdA, unlabel_train_idx, dataY, belief_threshold, step_size );
end

function [rank, threshold] = getData( wb, dataU, dataY, unlabel_train_idx)
    [num_unlabel,~] = size(dataU);
    dataUY = dataY(:,unlabel_train_idx);
    dataUY(dataUY~=0) = nan;
    dataUY(dataUY==0) = 1;
    outputs = wb * dataU';
    rank = outputs(1:end-1,:);
    threshold = outputs(end,:);
end

function [apd_train_idx, apd_rel_label, apd_irr_label, apd_penalty] = getAppendData( outputs, threshold, unlabel_train_idx, dataY, belief_threshold, step_size )
    [num_class,~] = size(outputs);
    outputs_t = outputs - repmat(threshold, num_class, 1);
    outputs_rel = outputs_t(:,:);
    outputs_irr = outputs_t(:,:);
    outputs_rel(outputs_rel<0) = nan;
    outputs_irr(outputs_irr>=0) = nan;
    [relValue, rellabel] = min(outputs_rel);
    [irrValue, irrlabel] = max(outputs_irr);
    diff = relValue - irrValue;

    idxinUA = find(diff > belief_threshold);
    [~, idxindiff] = topN(diff(idxinUA), min(step_size,length(idxinUA)), 'descend');
    idxinUA = idxinUA(idxindiff);
    virtual_label = ones(1,length(idxinUA))*(num_class+1);
    apd_train_idx = unlabel_train_idx(idxinUA);

    apd_rel_label = zeros(1,length(idxinUA));
    apd_irr_label = zeros(1,length(idxinUA));
    relValue = zeros(1,length(idxinUA));
    irrValue = zeros(1,length(idxinUA));
    for i = 1:length(idxinUA)
        candidates = find(~isnan(outputs_rel(:,idxinUA(i))));
        apd_rel_label(i) = candidates(randi(length(candidates)));
        relValue(i) = outputs_rel(apd_rel_label(i),idxinUA(i));
        candidates = find(~isnan(outputs_irr(:,idxinUA(i))));
        apd_irr_label(i) = candidates(randi(length(candidates)));
        irrValue(i) = outputs_irr(apd_irr_label(i),idxinUA(i));
    end

    apd_train_idx = [apd_train_idx,apd_train_idx,apd_train_idx];
    apd_rel_label = [apd_rel_label,apd_rel_label,virtual_label];
    apd_irr_label = [apd_irr_label,virtual_label,apd_irr_label];
    diff = [relValue - irrValue, relValue, -irrValue];
    belief = 1./(1+exp(-diff));
    apd_penalty = belief;
end
