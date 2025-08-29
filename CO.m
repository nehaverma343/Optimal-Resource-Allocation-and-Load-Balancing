function [BestCost_final, Convergence_curve, BestPosition, runtime] = CO(Function_name, D, SearchAgents_no, MaxFEs)
% Cheetah Optimization (CO)
% Inputs:
%   Function_name     - Name of benchmark function (e.g., 'F1')
%   D                 - Number of decision variables
%   SearchAgents_no   - Population size
%   MaxFEs            - Maximum number of function evaluations
% Outputs:
%   BestCost_final    - Best fitness value found
%   Convergence_curve - Fitness value at each iteration
%   BestPosition      - Best solution vector
%   runtime           - Total execution time

% --- Initialization ---
[lb, ub, ~, fobj] = Get_Functions_details(Function_name);
if length(lb) == 1
    lb = lb * ones(1, D);
    ub = ub * ones(1, D);
end

n = SearchAgents_no;
m = 2; % Number of agents in a group
T = ceil(D / 10) * 60;

empty_individual.Position = [];
empty_individual.Cost = [];
BestSol.Cost = inf;
pop = repmat(empty_individual, n, 1);

for i = 1:n
    pop(i).Position = lb + rand(1, D) .* (ub - lb);
    pop(i).Cost = fobj(pop(i).Position);
    if pop(i).Cost < BestSol.Cost
        BestSol = pop(i);
    end
end

pop1 = pop;
X_best = BestSol;
BestCost = [];
Globest = [];
FEs = 0;
t = 0;
it = 1;

Convergence_curve = [];

% --- Main Loop ---
tic;
while FEs <= MaxFEs
    i0 = randi(n, 1, m);
    for k = 1:m
        i = i0(k);
        a = i0(mod(k, m) + 1);

        X = pop(i).Position;
        X1 = pop(a).Position;
        Xb = BestSol.Position;
        Xbest = X_best.Position;

        kk = 0;
        if i <= 2 && t > 2 && t > ceil(0.2 * T + 1) && abs(BestCost(t - 2) - BestCost(t - ceil(0.2 * T + 1))) <= 0.0001 * Globest(t - 1)
            X = X_best.Position;
            kk = 0;
        elseif i == 3
            X = BestSol.Position;
            kk = -0.1 * rand * t / T;
        else
            kk = 0.25;
        end

        if mod(it, 100) == 0 || it == 1
            xd = randperm(D);
        end
        Z = X;

        for j = xd
            r_Hat = randn;
            r1 = rand;
            if k == 1
                alpha = 0.0001 * t / T * (ub(j) - lb(j));
            else
                alpha = 0.0001 * t / T * abs(Xb(j) - X(j)) + 0.001 * round(rand > 0.9);
            end

            r = randn;
            r_Check = abs(r)^exp(r / 2) * sin(2 * pi * r);
            beta = X1(j) - X(j);
            h0 = exp(2 - 2 * t / T);
            H = abs(2 * r1 * h0 - h0);

            r2 = rand;
            r3 = kk + rand;

            if r2 <= r3
                r4 = 3 * rand;
                if H > r4
                    Z(j) = X(j) + r_Hat^-1 * alpha;
                else
                    Z(j) = Xbest(j) + r_Check * beta;
                end
            else
                Z(j) = X(j);
            end
        end

        Z = min(max(Z, lb), ub);
        NewSol.Position = Z;
        NewSol.Cost = fobj(Z);
        FEs = FEs + 1;

        if NewSol.Cost < pop(i).Cost
            pop(i) = NewSol;
            if NewSol.Cost < BestSol.Cost
                BestSol = NewSol;
            end
        end
    end

    t = t + 1;

    if t > T && t - round(T) - 1 >= 1 && t > 2
        if abs(BestCost(t - 1) - BestCost(t - round(T) - 1)) <= abs(0.01 * BestCost(t - 1))
            best = X_best.Position;
            j0 = randi(D, 1, ceil(D / 10 * rand));
            best(j0) = lb(j0) + rand(1, length(j0)) .* (ub(j0) - lb(j0));
            BestSol.Position = best;
            BestSol.Cost = fobj(best);
            FEs = FEs + 1;

            i0 = randi(n, 1, n);
            pop(i0(n - m + 1:n)) = pop1(i0(1:m));
            pop(i) = X_best;
            t = 1;
        end
    end

    it = it + 1;

    if BestSol.Cost < X_best.Cost
        X_best = BestSol;
    end

    BestCost(t) = BestSol.Cost;
    Globest(t) = X_best.Cost;
    Convergence_curve(t) = X_best.Cost;

    if mod(it, 500) == 0
        disp(['FEs >> ' num2str(FEs) '   BestCost = ' num2str(X_best.Cost)]);
    end
end
runtime = toc;
BestCost_final = X_best.Cost;
BestPosition = X_best.Position;
end
