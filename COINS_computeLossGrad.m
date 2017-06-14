function [loss,gradient]= COINS_computeLossGrad(wb, dataX, dataY, pair_sizeA, train_idx_A, rel_label_A, irr_label_A, beliefA,...
    pair_sizeB, train_idx_B, rel_label_B, irr_label_B, beliefB, unlabel_train_idx, lambda, chl, epsilon, alpha, c, alpha2, c2)
%
% INPUT:
% wb                    (nC, 2*dim+2)   weight and bias of two views, [wbA, wbB]
% label_train_data      (nLT, dim)  labeled training data
% label_train_target    (nC, nLT)   labels of training data
% unlabel_train_data    (nU, dim)   unlabeled training data
% lambda    l2 regularizer for rank_svm
% epsilon   hyperparameter for balcan's epsilon expanding theory
% alpha, c, alpha2, c2  lagrangian multipliers for the augmented lagrangian methods
%
% OUTPUT:
% loss      the last loss
% gradient      (nC, 2*dim+2)   gradient
%
% VARIABLES:
% dataX     (nLT+nU, dim)   all the inputs
% dataY     (nC, nLT+nU)    labels for all the inputs

% INPUT:
% w 2*d weight vector
% xTrA    dx(nA) the training set for the first classifier
% yTrA    1x(nA) the labels
% beliefA 1x(nA) the weights
% xTrB    dx(nB) the training set for the second classifier
% yTrB    1x(nB) the labels
% beliefB 1x(nB) the weights
% lambda  l2 regularizer for logistic regression
% xU      dx(nU) the unlabeled set
% epsilon hyperparameter for balcan's epsilon expanding theory
% alpha, c, alpha2, c2    lagrangian multipliers for the augmented lagrangian methods
%

    dim = size(dataX, 2);
    num_class = size(wb, 1);

    wb1 = wb(:,1:dim+1);
    wb2 = wb(:,dim+2:end);
    w1 = wb(:,1:dim);
    w2 = wb(:,dim+2:end-1);

    lambda1 = dim/num_class;
    overlap = sum(sum(w1.*w1.*w2.*w2))*lambda1;
%     overlap = mean(mean(w1.*w1.*w2.*w2));
    loss = alpha*overlap + c/2*overlap*overlap;
%     gradient = [w1.*w2.*w2, zeros(num_class,1), w2.*w1.*w1, zeros(num_class,1)]*(alpha + c*overlap);
    gradient = 2*lambda1*[w1.*w2.*w2, zeros(num_class,1), w2.*w1.*w1, zeros(num_class,1)]*(alpha + c*overlap);

    view = ones(num_class, dim);
    [loss1, grad1] = COINS_RankingLoss( wb1, dataX, pair_sizeA, train_idx_A, rel_label_A, irr_label_A, view, beliefA, lambda, chl);
    [loss2, grad2] = COINS_RankingLoss( wb2, dataX, pair_sizeB, train_idx_B, rel_label_B, irr_label_B, view, beliefB, lambda, chl);

    if (loss1 > loss2)
        loss = loss + loss1 + log(1+exp(loss2-loss1));
    else
        loss = loss + loss2 + log(1+exp(loss1-loss2));
    end

    gradient = gradient + [1/(1+exp(loss2-loss1))*grad1, 1/(exp(loss1-loss2)+1)*grad2];

    [l, grad] = COINS_AllPairExpansion(wb, dataX(unlabel_train_idx,:), epsilon);
    loss = loss + alpha2*l + c2/2*l*l;
    gradient = gradient + grad*(alpha2+c2*l);
end
