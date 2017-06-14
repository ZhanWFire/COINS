function [viewA, viewB, w, l, converged] = COINS_feature_split(wb_last, dataX, dataY, pair_sizeA, train_idx_A, rel_label_A, irr_label_A, beliefA,...
    pair_sizeB, train_idx_B, rel_label_B, irr_label_B, beliefB,  maxIter, lambda, chl, unlabel_train_idx, epsilon)
%(w_last, xTrA, yTrA, beliefA, xTrB, yTrB, beliefB,  maxIter, lambda, xU, epsilon)
% INPUT:
% wb_last               (nC, 2*dim+2)   weight and bias of two views, [wbA, wbB]
% label_train_data      (nLT, dim)  labeled training data
% label_train_target    (nC, nLT)   labels of training data
% unlabel_train_data    (nU, dim)   unlabeled training data
% pos, neg number of positive/negative examples to add to the labeled set in each round
% lambda   l2 regularizer
% maxIter  maximum number of iterations of co-training
% epsilon  epsilon expanding condition
%
% OUTPUT:
% viewA, viewB  indices of view A and view B
% w             weight
% l             loss
%
% VARIABLES:
% dataX     (nLT+nU, dim)   all the inputs
% dataY     (nC, nLT+nU)    labels for all the inputs

    dim = size(dataX, 2);
    dataU = dataX(unlabel_train_idx,:);
    [num_class,~] = size(wb_last);
    % initial = w_last;
    lambda1 = dim/num_class;
    alpha = 0;
    c = 1.0;
    alpha2 = 0;
    c2 = 1.0;
    schedule = 2;
    converged = false;
    w1 = wb_last(:,1:dim);
    w2 = wb_last(:,dim+2:end-1);
    old_overlap = sum(sum(w1.*w1.*w2.*w2))*lambda1;
%     old_overlap = mean(mean(w1.*w1.*w2.*w2));
    old_expansion = COINS_AllPairExpansion(wb_last, dataU, epsilon);

    iter = 0;
    while c < 100
        iter = iter +1;
        [w,loss]= minimize(wb_last, @COINS_computeLossGrad, maxIter, dataX, dataY, pair_sizeA, train_idx_A, rel_label_A, irr_label_A, beliefA,...
            pair_sizeB, train_idx_B, rel_label_B, irr_label_B, beliefB, unlabel_train_idx, lambda, chl, epsilon, alpha, c, alpha2, c2);
        w1 = w(:,1:dim);
        w2 = w(:,dim+2:end-1);
        overlap = sum(sum(w1.*w1.*w2.*w2))*lambda1;
%         overlap = mean(mean(w1.*w1.*w2.*w2));

        alpha = alpha + c*overlap;
        if overlap >= 0.25*old_overlap
            c = c*schedule;
        end

        expansion = COINS_AllPairExpansion(w, dataU, epsilon);
        alpha2 = alpha2 + c2*expansion;
        if expansion >= 0.25*old_expansion
            c2 = c2*schedule;
        end
        if (overlap < 1e-3) && (expansion < 1e-3)
            converged = true;
            %['converged with c = ', num2str(c), ' alpha = ', num2str(alpha), ' c2 = ', num2str(c2), ' alpha2 = ', num2str(alpha2), ' at iter ', int2str(iter)]
            break;
        end
        wb_last = w;
        old_overlap = overlap;
        old_expansion = expansion;
    end
    l = loss(end);
    w1 = w(:,1:dim);
    w2 = w(:,dim+2:end-1);
    viewA = abs(w1) > abs(w2);
    viewB = abs(w2) > abs(w1);
end