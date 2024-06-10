 function optimized_k = apso_optimization(train_features, train_labels, k_range)
    % train_features: ѵ�����ݼ�������������ÿ��һ������������������
    % train_labels: ѵ�����ݼ��ı�ǩ��ÿ��Ԫ�ض�Ӧһ�������ı�ǩ��
    % k_range: Kֵ�ķ�Χ������ [1, 10]��
    
    % ����APSO�㷨�Ĳ���
    num_particles = 20; % ��������
    num_iterations = 50; % ��������
    w = 0.5; % ����Ȩ��
    c1 = 1; % ѧϰ����1
    c2 = 2; % ѧϰ����2
    
    % ��ʼ������Ⱥ��λ�ú��ٶ�
    num_dimensions = numel(k_range);
    particles_position = rand(num_particles, num_dimensions) .* repmat((k_range(:, 2) - k_range(:, 1)), [num_particles, 1]) + repmat(k_range(:, 1)', [num_particles, 1]);
    particles_velocity = rand(num_particles, num_dimensions) .* (k_range(:, 2) - k_range(:, 1));
    
    % ��ʼ������Ⱥ�ĸ������λ�ú���Ӧ��ֵ
    particles_best_position = particles_position;
    particles_best_fitness = inf(num_particles, 1);
    
    % ��ʼ��ȫ�����λ�ú���Ӧ��ֵ
    global_best_position = zeros(1, num_dimensions);
    global_best_fitness = inf;
    
    % ��ʼ����
    for iter = 1:num_iterations
        % ��������Ⱥÿ�����ӵ���Ӧ��ֵ
        for i = 1:num_particles
            fitness = knn_cross_validation(train_features, train_labels, particles_position(i, :));
            
            % ���¸������λ�ú���Ӧ��ֵ
            if fitness < particles_best_fitness(i)
                particles_best_fitness(i) = fitness;
                particles_best_position(i, :) = particles_position(i, :);
            end
            
            % ����ȫ�����λ�ú���Ӧ��ֵ
            if fitness < global_best_fitness
                global_best_fitness = fitness;
                global_best_position = particles_position(i, :);
            end
        end
        
        % ��������Ⱥ���ٶȺ�λ��
        r1 = rand(num_particles, num_dimensions);
        r2 = rand(num_particles, num_dimensions);
        particles_velocity = w .* particles_velocity + c1 .* r1 .* (particles_best_position - particles_position) + c2 .* r2 .* (repmat(global_best_position, [num_particles, 1]) - particles_position);
        particles_position = particles_position + particles_velocity;
        
        % ��������Ⱥ��λ����ָ����Χ��
        particles_position = max(particles_position, repmat(k_range(:, 1)', [num_particles, 1]));
        particles_position = min(particles_position, repmat(k_range(:, 2)', [num_particles, 1]));
    end
    
    % �����Ż����Kֵ
    optimized_k = round(global_best_position);
end
