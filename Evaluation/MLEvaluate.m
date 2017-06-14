function [ hammingLoss, rankingLoss, oneError, coverage, averagePrecision, macroAUC ] = MLEvaluate( outputs, pre_Labels,test_target )
%MLEvaluate evaluate the multi-label classifier
% outputs:      (nC, nTr)   the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
% pre_Labels:   (nC, nTr)   the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(j,i)=1, otherwise Pre_Labels(j,i)=-1
% test_target:  (nC, nTr)   the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    hammingLoss = HammingLoss(pre_Labels,test_target);
    rankingLoss = RankingLoss(outputs,test_target);
    oneError = OneError(outputs,test_target);
    coverage = Coverage(outputs,test_target);
    averagePrecision = AveragePrecision(outputs,test_target);
    macroAUC = MacroAUC(outputs,test_target);
end
