function [Best_score, WOA_curve, Best_pos, runtime] = PROPOSED(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)

lowerbound = ones(1, dimension) .* lowerbound;
upperbound = ones(1, dimension) .* upperbound;

% --- Initialization ---
X = zeros(SearchAgents, dimension);
fit = zeros(SearchAgents, 1);
for i = 1:dimension
    X(:, i) = lowerbound(i) + rand(SearchAgents, 1) .* (upperbound(i) - lowerbound(i));
end
for i = 1:SearchAgents
    fit(i) = fitness(X(i, :));
end

WOA_curve = zeros(Max_iterations, 1);
average = zeros(Max_iterations, 1);

% --- Main Loop ---
tic;
for t = 1:Max_iterations
    % --- Update best candidate ---
    [best, location] = min(fit);
    if t == 1
        Xbest = X(location, :);
        fbest = best;
    elseif best < fbest
        fbest = best;
        Xbest = X(location, :);
    end
    SW = Xbest;

    for i = 1:SearchAgents
        % --- Conditional Phase Selection ---
        if mod(t, 5) == 0 && mod(t, 7) == 0 && mod(t, 13) == 0
            % --- PHASE 1: FEEDING STRATEGY (EXPLORATION) ---
            I = round(1 + rand);
            X_P1 = X(i, :) + rand(1, dimension) .* (SW - I * X(i, :));
            X_P1 = min(max(X_P1, lowerbound), upperbound);
            F_P1 = fitness(X_P1);
            if F_P1 < fit(i)
                X(i, :) = X_P1;
                fit(i) = F_P1;
            end
        else
            % --- PHASE 3: ESCAPING AND FIGHTING AGAINST PREDATORS (EXPLOITATION) ---
            LO_LOCAL = lowerbound ./ t;
            HI_LOCAL = upperbound ./ t;
            I = round(1 + rand);
            X_P3 = X(i, :) + LO_LOCAL + rand * (HI_LOCAL - LO_LOCAL);
            X_P3 = min(max(X_P3, LO_LOCAL), HI_LOCAL);
            X_P3 = min(max(X_P3, lowerbound), upperbound);
            F_P3 = fitness(X_P3);
            if F_P3 < fit(i)
                X(i, :) = X_P3;
                fit(i) = F_P3;
            end
        end

        % --- PHASE 2: MIGRATION (Always Executed) ---
        I = round(1 + rand);
        K = randperm(SearchAgents); K(K == i) = [];
        X_K = X(K(1), :);
        F_RAND = fit(K(1));
        if fit(i) > F_RAND
            X_P2 = X(i, :) + rand * (X_K - I * X(i, :));
        else
            X_P2 = X(i, :) + rand * (X(i, :) - X_K);
        end
        X_P2 = min(max(X_P2, lowerbound), upperbound);
        F_P2 = fitness(X_P2);
        if F_P2 < fit(i)
            X(i, :) = X_P2;
            fit(i) = F_P2;
        end
    end

    WOA_curve(t) = fbest;
    average(t) = mean(fit);
end
runtime = toc;

Best_score = fbest;
Best_pos = Xbest;
end
