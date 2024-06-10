function predicted_labels = knn_classification(LBP_face, number_label, test_features, k)
    % LBP_face: ѵ�����ݼ�������������ÿ��һ������������������
    % number_label: ѵ�����ݼ��ı�ǩ��ÿ��Ԫ�ض�Ӧһ�������ı�ǩ��
    % test_features: �������ݼ�������������ÿ��һ������������������
    % k: KNN �е� K ֵ
    
    % ��ȡѵ�����ݼ��Ͳ������ݼ���������
    num_train_samples = size(LBP_face, 2);
    num_test_samples = size(test_features, 2);
    
    % ��ʼ��Ԥ���ǩ�ľ���
    predicted_labels = zeros(num_test_samples, 1);
    
    % ���ڲ������ݼ��е�ÿ������
    for i = 1:num_test_samples
        % �����������������ѵ������֮��ľ���
        distances = sqrt(sum((LBP_face - test_features(:, i)).^2, 1));
        
        % �ҵ���������� K ��ѵ������������
        [~, nearest_indices] = mink(distances, k);
        
        % ���� K ������ھӵı�ǩ����ͶƱ
        nearest_labels = number_label(nearest_indices);
        predicted_labels(i) = mode(nearest_labels);
    end
end
