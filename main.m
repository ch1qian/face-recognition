% 主程序
function main()
    % 从训练图像库中加载训练数据和标签
    load('fb_lbp_face.mat');
    
    % 从验证图像库中加载验证数据和标签
    load('fa_lbp_face.mat');
    
    % 应用APSO-KNN算法获取最优K值
    k_range = 1:10; % K值的范围
    folds = 5; % 交叉验证的折数
    optimized_k = apso_knn_optimization(LBP_face, number_label, k_range, folds);
    
    % 使用最优K值进行KNN分类
    [classification_result, ~] = knn_classification(LBP_face, number_label, fa_face, optimized_k);
    
    % 显示分类结果
    display_result(fa_face, classification_result);
end

% 显示分类结果
function display_result(images, results)
    % 创建新的图像窗口
    figure;
    
    % 循环显示每张图像及其分类结果
    for i = 1:length(images)
        % 将图像显示在第一个子图中
        subplot(length(images), 2, 2*i-1);
        imshow(images{i});
        title('Original Image');
        
        % 将分类结果显示在第二个子图中
        subplot(length(images), 2, 2*i);
        imshow(results{i});
        title('Classification Result');
    end
end
