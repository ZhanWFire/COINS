function [coverage]=Coverage(Outputs,test_target)
%Computing the coverage
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance]=size(Outputs);
    cov = 0;
    for i=1:num_instance
        temp=Outputs(:,i);
        [~,rank] = sort(temp,'descend');
        rank(rank) = 1:num_class;
        cov = cov + max([rank(test_target(:,i)==1);0]);
    end
    coverage=(cov/num_instance)-1;
end