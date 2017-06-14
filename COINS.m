function [wb, loss, acc, iter_cnt]  = COINS(label_train_data, label_train_target, unlabel_train_data, chl, maxIter, epsilon,...
    test_data, test_target, unlabel_target, is_debug, draw_pic)
% INPUT:
% label_train_data      (nLT, dim)  labeled training data
% label_train_target    (nC, nLT)   labels of training data
% unlabel_train_data    (nU, dim)   unlabeled training data
% pos, neg number of positive/negative examples to add to the labeled set in each round
% lambda   l2 regularizer
% maxIter  maximum number of iterations of co-training
% epsilon  epsilon expanding condition
%
% OUTPUT:
% weights   all the weights each iteration
% loss      the last loss
% acc       accuracy each iteration
%
% VARIABLES:
% dataX     (nLT+nU, dim)   all the inputs
% dataY     (nC, nLT+nU)    labels for all the inputs

    [label_train_data, label_train_target, label_train_idx, rel_label, irr_label] = COINS_ProcessInput(label_train_data, label_train_target);
    [num_label_train, dim] = size(label_train_data);
    num_unlabel_train = size(unlabel_train_data,1);
    num_class = size(label_train_target,1);

    dataX = [label_train_data; unlabel_train_data];
    dataY = [label_train_target, zeros(num_class, num_unlabel_train)];

    unlabel_train_idx = num_label_train+1 : num_label_train + num_unlabel_train;

    train_idx_A = label_train_idx;
    rel_label_A = rel_label;
    irr_label_A = irr_label;
    train_idx_B = label_train_idx;
    rel_label_B = rel_label;
    irr_label_B = irr_label;

    penalty_A = ones(1, length(train_idx_A));
    penalty_B = ones(1, length(train_idx_B));

%     weights = [];
    iter = 1;
    iter_cnt = 20;
    lambda = 1/num_class;
    acc = zeros(maxIter, 3, 6);
    wb_last = [getRandInitWeight(num_class+1,dim+1), getRandInitWeight(num_class+1,dim+1)];%randn(num_class,2*dim+2);
    pair_size = hist(label_train_idx, 1:num_label_train + num_unlabel_train);
    pair_sizeA = pair_size;
    pair_sizeB = pair_size;

    while size(unlabel_train_idx,2) >= 0 && iter <= maxIter
%         assert( sum(dataY(idxLabs) ~= labels(idxLabs)) == 0);
        bA = penalty_A;%/length(train_idx_A)/2;
        bB = penalty_B;%/length(train_idx_B)/2;

        converged = false;
        while true
            wb_last = [getRandInitWeight(num_class+1,dim+1), getRandInitWeight(num_class+1,dim+1)];
            [viewA, viewB, wb, loss_split, converged] = ...
                COINS_feature_split(wb_last, dataX, dataY, pair_sizeA, train_idx_A, rel_label_A, irr_label_A, bA,...
                            pair_sizeB, train_idx_B, rel_label_B, irr_label_B, bB, iter_cnt, lambda, chl, unlabel_train_idx, epsilon);
            if min(sum(viewA,2).*sum(viewB,2))>0
                break;
            end
%             wb_last = [getRandInitWeight(num_class+1,dim+1), getRandInitWeight(num_class+1,dim+1)];
        end
        disp(['iter = ',int2str(iter),', converged = ',int2str(converged)]);
        w1 = wb(:,1:dim);
        w2 = wb(:,dim+2:end-1);

        initial = zeros(num_class+1, dim+1);
        [new_wbA, lossA]= minimize([w1.*viewA,wb(:,dim+1)], @COINS_RankingLoss, iter_cnt, dataX, pair_sizeA, train_idx_A, rel_label_A, irr_label_A, viewA, bA, lambda, chl);
        [new_wbB, lossB]= minimize([w2.*viewB,wb(:,end)], @COINS_RankingLoss, iter_cnt, dataX, pair_sizeB, train_idx_B, rel_label_B, irr_label_B, viewB, bB, lambda, chl);

        wb = [new_wbA, new_wbB];
        wb_last = wb;

        if is_debug
            [acc_temp] = getAcc(wb, test_data, test_target);
            acc(iter,:,:) = acc_temp;
            disp('acc.');
            disp(reshape(acc(iter,1:3,:),3,6));
        end

        if draw_pic
            evaluate_method = {'hammingLoss', 'rankingLoss', 'oneError', 'coverage', 'averagePrecision', 'macroAUC'};
            for i=1:6
                subplot(2,3,i);
                plot(acc(1:iter,1:3,i));
                title(evaluate_method{i});
                drawnow;
            end
        end

        find_apd_max_iter = 30;
        step_size = max(5,ceil(num_label_train/10));
        fprintf('B:\n');
        [is_overB, train_idx_B, rel_label_B, irr_label_B, penalty_B, pair_sizeB, used_idxB] = ...
            appendOneView(new_wbA, new_wbB, dataX, dataY, unlabel_train_idx, find_apd_max_iter, train_idx_B, rel_label_B, irr_label_B, penalty_B, viewB,...
            lambda, chl, initial, iter_cnt, step_size, pair_sizeB, label_train_data, label_train_target, num_label_train, unlabel_target, is_debug);
        fprintf('A:\n');
        [is_overA, train_idx_A, rel_label_A, irr_label_A, penalty_A, pair_sizeA, used_idxA] = ...
            appendOneView(new_wbB, new_wbA, dataX, dataY, unlabel_train_idx, find_apd_max_iter, train_idx_A, rel_label_A, irr_label_A, penalty_A, viewA,...
            lambda, chl, initial, iter_cnt, step_size, pair_sizeA, label_train_data, label_train_target, num_label_train, unlabel_target, is_debug);

        if is_overA && is_overB
            break;
        end

        unlabel_train_idx = setdiff(unlabel_train_idx, union(used_idxB, used_idxA));
        iter = iter+1;
    end
    loss = loss_split;
    iter_cnt = iter - 1;
end

function [initial] = getRandInitWeight(a,b)
    initial = randn(a,b);
    for i = 1:a
        initial(i,:) = initial(i,:)/norm(initial(i,:));
    end
end

function [is_overB, train_idx_B, rel_label_B, irr_label_B, penalty_B, pair_sizeB, used_idxB] = ...
    appendOneView(new_wbA, new_wbB, dataX, dataY, unlabel_train_idx, find_apd_max_iter, train_idx_B, rel_label_B, irr_label_B, penalty_B, viewB,...
    lambda, chl, initial, iter_cnt, step_size, pair_sizeB, label_train_data, label_train_target, num_label_train, unlabel_target, is_debug)

    [confidenceB, pre_labelsB] = COINS_predict_one_view(new_wbB, label_train_data);
    hlB = RankingLoss(confidenceB, label_train_target);
    is_overB = true;
    used_idxB= [];
    for i = 1:find_apd_max_iter
        [apd_train_idx_B, apd_rel_label_B, apd_irr_label_B, apd_penalty_B ] = ...
            COINS_AppendTrainingSet( new_wbA, dataX, dataY, unlabel_train_idx, step_size );
        if isempty(apd_train_idx_B)
            fprintf('empty.');
            is_overB = true;
            break;
        end

        train_idx_B_new = [train_idx_B, apd_train_idx_B];
        rel_label_B_new = [rel_label_B, apd_rel_label_B];
        irr_label_B_new = [irr_label_B, apd_irr_label_B];
        penalty_B_new = [penalty_B, apd_penalty_B];

        pair_size_new = pair_sizeB + hist(apd_train_idx_B, 1:length(pair_sizeB));
        [temp_wbB, lossB]= minimize(initial, @COINS_RankingLoss, iter_cnt, dataX, pair_size_new, ...
            train_idx_B_new, rel_label_B_new, irr_label_B_new, viewB, penalty_B_new, lambda, chl);
        [confidenceB, pre_labelsB] = COINS_predict_one_view(temp_wbB, label_train_data);
        hlB_new = RankingLoss(confidenceB, label_train_target);

        if is_debug
            [total_loss, rel_loss, irr_loss, ts_loss, total_num, p_rel_loss, p_irr_loss, p_ts_loss] = ...
                getApdAcc(num_label_train, unlabel_target, apd_train_idx_B, apd_rel_label_B, apd_irr_label_B);
            fprintf('%d \t%d \t%d \t%d \t%d  | \t%d \t%d \t%d \t%d\n', ...
                total_loss, rel_loss, irr_loss, ts_loss, total_num, p_rel_loss, p_irr_loss, p_ts_loss, total_num/3);
            fprintf('rl: %4.4f \t%4.4f\n',hlB_new,hlB);
        end

        if hlB_new < hlB
            used_idxB = unique(apd_train_idx_B);
            is_overB = false;
            break;
        end
    end

    if ~is_overB
        train_idx_B = train_idx_B_new;
        rel_label_B = rel_label_B_new;
        irr_label_B = irr_label_B_new;
        penalty_B = penalty_B_new;
        pair_sizeB = pair_size_new;
        fprintf('i: %d, rl: %4.4f \t%4.4f\n',i,hlB_new,hlB);
    else
        fprintf('i: %d, none.\n',i);
    end
end

function [total_loss, rel_loss, irr_loss, ts_loss, total_num, p_rel_loss, p_irr_loss, p_ts_loss] = ...
    getApdAcc(num_label_train, unlabel_target, apd_train_idx, apd_rel_label, apd_irr_label)

    [num_class,~] = size(unlabel_target);
    total_num = length(apd_train_idx);
    total_loss = 0;
    rel_loss = 0;
    irr_loss = 0;
    ts_loss = 0;
    p_rel_loss = 0;
    p_irr_loss = 0;
    p_ts_loss = 0;
    for i = 1 : total_num
        idx = apd_train_idx(i)-num_label_train;
        if apd_rel_label(i) == num_class+1
            irr = unlabel_target( apd_irr_label(i), idx ) ~= -1;
            rel = irr;
        else
            if apd_irr_label(i) == num_class+1
                rel = unlabel_target( apd_rel_label(i), idx ) ~= 1;
                irr = rel;
            else
                rel = unlabel_target( apd_rel_label(i), idx ) ~= 1;
                irr = unlabel_target( apd_irr_label(i), idx ) ~= -1;
                p_rel_loss = p_rel_loss + rel;
                p_irr_loss = p_irr_loss + irr;
                p_ts_loss = p_ts_loss + (rel && irr);
            end
        end
        total_loss = total_loss + rel + irr;
        rel_loss = rel_loss + rel;
        irr_loss = irr_loss + irr;
        ts_loss = ts_loss + (rel && irr);
    end
end

function [acc] = getAcc(wb, test_data, test_target)
    acc = zeros(3,6);
    [ confidence, pre_labels, confidenceA, confidenceB ] = COINS_predict( wb, test_data );

    [ hl1, rl1, oe1, cov1, ap1, auc1 ] = MLEvaluate(confidenceA, getPreLabel(confidenceA), test_target);
    [ hl2, rl2, oe2, cov2, ap2, auc2 ] = MLEvaluate(confidenceB, getPreLabel(confidenceB), test_target);
    [ hl3, rl3, oe3, cov3, ap3, auc3 ] = MLEvaluate(confidence, pre_labels, test_target);

    acc(1,:) = [ hl1, rl1, oe1, cov1, ap1, auc1 ];
    acc(2,:) = [ hl2, rl2, oe2, cov2, ap2, auc2 ];
    acc(3,:) = [ hl3, rl3, oe3, cov3, ap3, auc3 ];
end

function [pre_labels] = getPreLabel(confidence)
    pre_labels = zeros(size(confidence));
    pre_labels(confidence>=0.5)=1;
    pre_labels(confidence<0.5)=-1;
end

