function [macroAUC, auc] = MacroAUC(Outputs,test_target)
%Computing the Macro_AUC
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance]=size(Outputs);
    count = zeros(num_class,1);   % store the number of postive instance'score>negative instance'score
    num_P_instance = zeros(num_class,1);%number of positive instance for every label
    num_N_instance = zeros(num_class,1);
    auc = zeros(num_class,1);  % auc for each label
    count_valid_label = 0;

    for i = 1:num_class
        num_P_instance(i,1) = sum(test_target(i,:) == 1);
        num_N_instance(i,1) = num_instance - num_P_instance(i,1);
        % exclude the label on which all instances are positive or negative,
        % leading to num_P_instance(i,1) or num_N_instance(i,1) is zero
        if num_P_instance(i,1) == 0 || num_N_instance(i,1) == 0
            auc(i,1) = 0;
            count_valid_label = count_valid_label + 1;
        else
            temp_P_Outputs = zeros(num_class,num_P_instance(i,1));
            temp_N_Outputs = zeros(num_class,num_N_instance(i,1));

            temp_P_Outputs(i,:) = Outputs(i,test_target(i,:) == 1);
            temp_N_Outputs(i,:) = Outputs(i,test_target(i,:) == -1);

            for m = 1:num_P_instance(i,1)
                for n = 1: num_N_instance(i,1)
                    if(temp_P_Outputs(i,m) > temp_N_Outputs(i,n))
                        count(i,1) = count(i,1) + 1;
                    elseif(temp_P_Outputs(i,m) == temp_N_Outputs(i,n))
                        count(i,1) = count(i,1) + 0.5;
                    end
                end
            end
            auc(i,1) = count(i,1)/(num_P_instance(i,1)*num_N_instance(i,1));
        end
    end   
    macroAUC = sum(auc)/(num_class-count_valid_label);

end

