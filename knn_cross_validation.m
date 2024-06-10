


function accuracy = knn_cross_validation(train_features, train_labels, k)
    % train_features: ѵ�����ݼ�������������ÿ��һ������������������
    % train_labels: ѵ�����ݼ��ı�ǩ��ÿ��Ԫ�ض�Ӧһ�������ı�ǩ��
    % k: KNN�е�Kֵ
    
    % ���彻����֤�Ĳ���
    num_folds = 5; % ����
    num_samples = numel(train_labels);
    fold_size = floor(num_samples / num_folds);
    
    % ��ʼ��׼ȷ������
    accuracies = zeros(num_folds, 1);
    
    % ִ�н�����֤
    for fold = 1:num_folds
        % ����ѵ��������֤��
        start_idx = (fold - 1) * fold_size + 1;
        end_idx = fold * fold_size;
        if fold == num_folds
            end_idx = num_samples;
        end
        
        % ��ȡ��֤���������ͱ�ǩ
        validation_features = train_features(start_idx:end_idx, :);
        validation_labels = train_labels(start_idx:end_idx);
        
        % ��ȡѵ�����������ͱ�ǩ
        train_features_fold = [train_features(1:start_idx-1, :); train_features(end_idx+1:end, :)];
        train_labels_fold = [train_labels(1:start_idx-1); train_labels(end_idx+1:end)];
        
        % ʹ��KNN�㷨���з���
        predicted_labels = knn_classification(train_features_fold, train_labels_fold, validation_features, k);
        
        % ������֤����׼ȷ��
        correct_predictions = sum(predicted_labels == validation_labels);
        accuracy = correct_predictions / numel(validation_labels);
        accuracies(fold) = accuracy;
    end
    
    % ����ƽ��׼ȷ��
    accuracy = mean(accuracies);
end
