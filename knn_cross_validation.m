


function accuracy = knn_cross_validation(train_features, train_labels, k)
    % train_features: 训练数据集的特征向量（每行一个样本的特征向量）
    % train_labels: 训练数据集的标签（每个元素对应一个样本的标签）
    % k: KNN中的K值
    
    % 定义交叉验证的参数
    num_folds = 5; % 折数
    num_samples = numel(train_labels);
    fold_size = floor(num_samples / num_folds);
    
    % 初始化准确率数组
    accuracies = zeros(num_folds, 1);
    
    % 执行交叉验证
    for fold = 1:num_folds
        % 划分训练集和验证集
        start_idx = (fold - 1) * fold_size + 1;
        end_idx = fold * fold_size;
        if fold == num_folds
            end_idx = num_samples;
        end
        
        % 获取验证集的样本和标签
        validation_features = train_features(start_idx:end_idx, :);
        validation_labels = train_labels(start_idx:end_idx);
        
        % 获取训练集的样本和标签
        train_features_fold = [train_features(1:start_idx-1, :); train_features(end_idx+1:end, :)];
        train_labels_fold = [train_labels(1:start_idx-1); train_labels(end_idx+1:end)];
        
        % 使用KNN算法进行分类
        predicted_labels = knn_classification(train_features_fold, train_labels_fold, validation_features, k);
        
        % 计算验证集的准确率
        correct_predictions = sum(predicted_labels == validation_labels);
        accuracy = correct_predictions / numel(validation_labels);
        accuracies(fold) = accuracy;
    end
    
    % 返回平均准确率
    accuracy = mean(accuracies);
end
