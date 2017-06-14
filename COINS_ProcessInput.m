function [ SVs, target, train_idx, rel_label, irr_label ] = COINS_ProcessInput( train_data, train_target )
%ProcessInput, process the dataset
%
% INPUT:
% train_data    (nLT, dim)      labeled training data
% train_target  (nL, nLT)       labels of training data
%
% OUTPUT:
% SVs           (nSVs, dim)     support vectors
% train_idx     (1, num_pairs)  index of training data
% rel_label     (1, num_pairs)  relevant label in every pair
% irr_label     (1, num_pairs)  irrelevant label in every pair

    num_class = size(train_target, 1);
    SVs_idx = abs(sum(train_target,1))~=num_class;
    SVs = train_data(SVs_idx,:);
    target = train_target(:,SVs_idx);

    num_training = size(SVs, 1);
    Label = cell(num_training,1);
    not_Label = cell(num_training,1);
    Label_size = zeros(1,num_training);
    size_alpha = zeros(1,num_training);
    for i=1:num_training
        temp = target(:,i);
        Label_size(1,i) = sum(temp==ones(num_class,1));
        size_alpha(1,i) = Label_size(1,i)*(num_class-Label_size(1,i));
        for j=1:num_class
            if(temp(j)==1)
                Label{i,1} = [Label{i,1},j];
            else
                not_Label{i,1} = [not_Label{i,1},j];
            end
        end
    end

    pair_size = sum(size_alpha);
    train_idx = zeros(1,pair_size);
    rel_label = zeros(1,pair_size);
    irr_label = zeros(1,pair_size);
%     penalty = zeros(1,pair_size);
    virtual_label = num_class + 1;

    count = 1;
    for i=1:num_training
        for m=1:Label_size(i)
            for n=1:(num_class-Label_size(i))
                train_idx(count) = i;
                rel_label(count) = Label{i,1}(m);
                irr_label(count) = not_Label{i,1}(n);
%                 penalty(count) = 1;
                count = count + 1;
            end
        end
        for j=1:num_class
            train_idx(count) = i;
            if target(j,i) == 1
                rel_label(count) = j;
                irr_label(count) = virtual_label;
%                 penalty(count) = 1.5;
            else
                rel_label(count) = virtual_label;
                irr_label(count) = j;
%                 penalty(count+1) = 1.5;
            end
            count = count + 1;
        end
    end

end

