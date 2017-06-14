function [rankingLoss]=RankingLoss(Outputs,test_target)
%Computing the hamming loss
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,~]=size(Outputs);
    index = (sum(test_target)~=num_class)&(sum(test_target)~=-num_class);
    temp_Outputs = Outputs(:,index);
    temp_test_target = test_target(:,index);
    [~,num_instance]=size(temp_Outputs);

    rl = 0;
    for i=1:num_instance
        res = bsxfun(@le,temp_Outputs(temp_test_target(:,i)>0,i),temp_Outputs(temp_test_target(:,i)<=0,i)');
        rl = rl + mean(mean(res));
    end
    rankingLoss = rl / num_instance;
end
