function [ loss, gradient ] = COINS_RankingLoss( wb, train_data, pair_size, train_idx, rel_label, irr_label, view, belief, lambda, chl)
%
% INPUT:
% wb            (nC+1, dim+1)     weight and bias vector
% train_data    (nTL, dim)      training data
% num_class     number of labels
% train_idx     (1, num_pairs)  index of training data
% rel_label     (1, num_pairs)  relevant label in every pair
% irr_label     (1, num_pairs)  irrelevant label in every pair
% view          (nC, dim)       view
% belief        (1, nTL)        matrix (weights for each input)
% lambda        l2 regularization factor
%
% OUTPUT:
% loss          loss
% gradient      (nC, dim+1)     gradient

    c = chl/length(unique(train_idx));

%     belief = ones(1,length(train_idx));
    [num_train_data, dim] = size(train_data);
    [num_class, ~] = size(wb);
    wb(:,1:dim) = wb(:,1:dim) .* view;
    w = wb(:,1:dim);
    train_data = [train_data, ones(num_train_data, 1)];

    loss = lambda/2 * sum(sum(w.*w));
    value = max(0, 1 - sum( (wb(rel_label,:)-wb(irr_label,:)) .* train_data(train_idx,:), 2 ) ) .* belief' ./ pair_size(train_idx)';
    hinge_loss = c * sum(value);%/length(train_idx);
    loss = loss + hinge_loss;

    vidx_0 = (value==0);
    rel_label(vidx_0) = 0;
    irr_label(vidx_0) = 0;
    gradient1 = lambda * [w, zeros(num_class,1)];
    gradient2 = zeros(num_class, dim+1);
    for i = 1:num_class
        rel_idx = train_idx(rel_label==i);
        irr_idx = train_idx(irr_label==i);
        gradient2(i,:) = gradient2(i,:) - sum(train_data(rel_idx,:).*repmat(belief(rel_label==i)'./pair_size(rel_idx)',1,dim+1)) ...
            + sum(train_data(irr_idx,:).*repmat(belief(irr_label==i)'./pair_size(irr_idx)',1,dim+1));
    end
    gradient = gradient1 + c * gradient2;
    gradient(:,1:end-1) = gradient(:,1:end-1).*view;
end

