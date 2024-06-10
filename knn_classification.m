function predicted_labels = knn_classification(LBP_face, number_label, test_features, k)
    % LBP_face: 训练数据集的特征向量（每列一个样本的特征向量）
    % number_label: 训练数据集的标签（每个元素对应一个样本的标签）
    % test_features: 测试数据集的特征向量（每列一个样本的特征向量）
    % k: KNN 中的 K 值
    
    % 获取训练数据集和测试数据集的样本数
    num_train_samples = size(LBP_face, 2);
    num_test_samples = size(test_features, 2);
    
    % 初始化预测标签的矩阵
    predicted_labels = zeros(num_test_samples, 1);
    
    % 对于测试数据集中的每个样本
    for i = 1:num_test_samples
        % 计算测试样本与所有训练样本之间的距离
        distances = sqrt(sum((LBP_face - test_features(:, i)).^2, 1));
        
        % 找到距离最近的 K 个训练样本的索引
        [~, nearest_indices] = mink(distances, k);
        
        % 根据 K 个最近邻居的标签进行投票
        nearest_labels = number_label(nearest_indices);
        predicted_labels(i) = mode(nearest_labels);
    end
end
