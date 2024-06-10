% ������
function main()
    % ��ѵ��ͼ����м���ѵ�����ݺͱ�ǩ
    load('fb_lbp_face.mat');
    
    % ����֤ͼ����м�����֤���ݺͱ�ǩ
    load('fa_lbp_face.mat');
    
    % Ӧ��APSO-KNN�㷨��ȡ����Kֵ
    k_range = 1:10; % Kֵ�ķ�Χ
    folds = 5; % ������֤������
    optimized_k = apso_knn_optimization(LBP_face, number_label, k_range, folds);
    
    % ʹ������Kֵ����KNN����
    [classification_result, ~] = knn_classification(LBP_face, number_label, fa_face, optimized_k);
    
    % ��ʾ������
    display_result(fa_face, classification_result);
end

% ��ʾ������
function display_result(images, results)
    % �����µ�ͼ�񴰿�
    figure;
    
    % ѭ����ʾÿ��ͼ���������
    for i = 1:length(images)
        % ��ͼ����ʾ�ڵ�һ����ͼ��
        subplot(length(images), 2, 2*i-1);
        imshow(images{i});
        title('Original Image');
        
        % ����������ʾ�ڵڶ�����ͼ��
        subplot(length(images), 2, 2*i);
        imshow(results{i});
        title('Classification Result');
    end
end
