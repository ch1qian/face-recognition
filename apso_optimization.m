 function optimized_k = apso_optimization(train_features, train_labels, k_range)
    % train_features: 训练数据集的特征向量（每行一个样本的特征向量）
    % train_labels: 训练数据集的标签（每个元素对应一个样本的标签）
    % k_range: K值的范围（例如 [1, 10]）
    
    % 定义APSO算法的参数
    num_particles = 20; % 粒子数量
    num_iterations = 50; % 迭代次数
    w = 0.5; % 惯性权重
    c1 = 1; % 学习因子1
    c2 = 2; % 学习因子2
    
    % 初始化粒子群的位置和速度
    num_dimensions = numel(k_range);
    particles_position = rand(num_particles, num_dimensions) .* repmat((k_range(:, 2) - k_range(:, 1)), [num_particles, 1]) + repmat(k_range(:, 1)', [num_particles, 1]);
    particles_velocity = rand(num_particles, num_dimensions) .* (k_range(:, 2) - k_range(:, 1));
    
    % 初始化粒子群的个体最佳位置和适应度值
    particles_best_position = particles_position;
    particles_best_fitness = inf(num_particles, 1);
    
    % 初始化全局最佳位置和适应度值
    global_best_position = zeros(1, num_dimensions);
    global_best_fitness = inf;
    
    % 开始迭代
    for iter = 1:num_iterations
        % 计算粒子群每个粒子的适应度值
        for i = 1:num_particles
            fitness = knn_cross_validation(train_features, train_labels, particles_position(i, :));
            
            % 更新个体最佳位置和适应度值
            if fitness < particles_best_fitness(i)
                particles_best_fitness(i) = fitness;
                particles_best_position(i, :) = particles_position(i, :);
            end
            
            % 更新全局最佳位置和适应度值
            if fitness < global_best_fitness
                global_best_fitness = fitness;
                global_best_position = particles_position(i, :);
            end
        end
        
        % 更新粒子群的速度和位置
        r1 = rand(num_particles, num_dimensions);
        r2 = rand(num_particles, num_dimensions);
        particles_velocity = w .* particles_velocity + c1 .* r1 .* (particles_best_position - particles_position) + c2 .* r2 .* (repmat(global_best_position, [num_particles, 1]) - particles_position);
        particles_position = particles_position + particles_velocity;
        
        % 限制粒子群的位置在指定范围内
        particles_position = max(particles_position, repmat(k_range(:, 1)', [num_particles, 1]));
        particles_position = min(particles_position, repmat(k_range(:, 2)', [num_particles, 1]));
    end
    
    % 返回优化后的K值
    optimized_k = round(global_best_position);
end
