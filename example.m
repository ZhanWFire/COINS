%This is an examplar file on how the COINS program could be used (The main function is "COINS.m")
%
%Type 'help COINS' under Matlab prompt for more detailed information
%
%N.B.: Please ensure that the code in the folder "Evaluation" is added to the matlab path or in the same path of COINS.
%


% Loading the file containing the necessary inputs for calling the COINS function
load('sample_data.mat');

%Set the experiment parameter
is_debug = true;
draw_pic = false;

%Set the ratio parameter used by COINS
chl = 1000;
epsilon = 10;
maxIter = 30;

% Calling the main function COINS
t0 = cputime;
[wb, loss, acc, iter]  = COINS(label_data, label_target, unlabel_data, chl, maxIter, epsilon, test_data, test_target, unlabel_target, is_debug, draw_pic);
cost_time = cputime - t0;
[ test_outputs, test_pre_labels ] = COINS_predict( wb, test_data );
[ unlabel_outputs, unlabel_pre_labels ] = COINS_predict( wb, unlabel_data );
[ hl1, rl1, oe1, cov1, ap1, auc1 ] = MLEvaluate(test_outputs, test_pre_labels, test_target);
[ hl2, rl2, oe2, cov2, ap2, auc2 ] = MLEvaluate(unlabel_outputs, unlabel_pre_labels, unlabel_target);
res = [ hl1, rl1, oe1, cov1, ap1, auc1;hl2, rl2, oe2, cov2, ap2, auc2 ];
disp(res);
