clc;clear;
load 'metrics\metrics_tst_base'
load 'metrics\metrics_tst_aug'
load 'metrics\metrics_new_base'
load 'metrics\metrics_new_aug'
%% IoU
[metrics_tst_base.ClassMetrics.IoU, metrics_tst_aug.ClassMetrics.IoU]
[mean(metrics_tst_base.ClassMetrics.IoU), ...
    mean(metrics_tst_aug.ClassMetrics.IoU)]

[metrics_new_base.ClassMetrics.IoU, metrics_new_aug.ClassMetrics.IoU]
[mean(metrics_new_base.ClassMetrics.IoU), ...
    mean(metrics_new_aug.ClassMetrics.IoU)]
%% Confusion matrix
metrics = {'metrics_tst_base', 'metrics_tst_aug',...
            'metrics_new_base', 'metrics_new_aug'};
        
        
for k=1:4
    cm = eval([metrics{k}, '.NormalizedConfusionMatrix']);
    var = cm.Variables;
    fprintf([metrics{k}, '\n'])
    fprintf(' Precision |   Recall | Accuracy |    F1    |\n')
    
    PRAF = zeros(4,4);
    
    for i=1:4
        target = i;
        TP = var(target,target);
        FN = sum(var(target,:)) - TP;
        FP = sum(var(:,target)) - TP;
        TN = sum(var) - FN - FP - TP;
        Precision = TP/(TP+FP);
        Recall = TP/(TP+FN);
        Accuracy = (TP+TN)/(TP+FP+TN+FN);
        F1 = 2 * (Precision * Recall)/(Precision + Recall);
        fprintf('  %8.4f | %8.4f | %8.4f | %8.4f |\n', ...
            Precision, Recall, Accuracy, F1)   
        PRAF(i,:) = [Precision, Recall, Accuracy, F1];
    end
    mean(PRAF,1)
end
