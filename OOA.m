function [fbest, Convergence_curve, best_so_far, runtime] = OOA(SearchAgents, fitness, lowerbound, upperbound, Max_iterations)
    % Ocean Optimization Algorithm (OOA) in MATLAB

    [N, dim] = size(SearchAgents);
    X = SearchAgents;
    fit = zeros(N, 1);

    for i = 1:N
        fit(i) = fitness(X(i, :)); % Initial fitness evaluation
    end

    Convergence_curve = zeros(Max_iterations, 1);
    best_so_far = zeros(Max_iterations, 1);
    average = zeros(Max_iterations, 1);

    tic;
    for t = 1:Max_iterations
        % --- Update best solution ---
        [Fbest, blocation] = min(fit);

        if t == 1
            xbest = X(blocation, :);
            fbest = Fbest;
        elseif Fbest < fbest
            fbest = Fbest;
            xbest = X(blocation, :);
        end

        for i = 1:N
            % --- Phase 1: Exploration ---
            fish_position = find(fit < fit(i)); % Eq(4)

            if isempty(fish_position)
                selected_fish = xbest;
            else
                if rand < 0.5
                    selected_fish = xbest;
                else
                    k = fish_position(randi(length(fish_position)));
                    selected_fish = X(k, :);
                end
            end

            I = round(1 + rand);
            X_new_P1 = X(i, :) + rand * (selected_fish - I * X(i, :)); % Eq(5)
            X_new_P1 = max(X_new_P1, lowerbound);
            X_new_P1 = min(X_new_P1, upperbound);

            fit_new_P1 = fitness(X_new_P1);
            if fit_new_P1 < fit(i)
                X(i, :) = X_new_P1;
                fit(i) = fit_new_P1;
            end

            % --- Phase 2: Exploitation ---
            X_new_P2 = X(i, :) + (lowerbound + rand(1, dim) .* (upperbound - lowerbound)) / t; % Eq(7)
            X_new_P2 = max(X_new_P2, lowerbound);
            X_new_P2 = min(X_new_P2, upperbound);

            fit_new_P2 = fitness(X_new_P2);
            if fit_new_P2 < fit(i)
                X(i, :) = X_new_P2;
                fit(i) = fit_new_P2;
            end
        end

        best_so_far(t) = fbest;
        average(t) = mean(fit);
        Convergence_curve(t) = fbest;
    end

    runtime = toc;
    best_so_far = Convergence_curve(end);
end
