function [y_global_best, Convergence_curve, x_global_best, runtime] = ESO(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

x = initialization(SearchAgents_no, dim, ub, lb);
w = rand(SearchAgents_no, dim) * 2 - 1;
m = zeros(SearchAgents_no, dim);
v = zeros(SearchAgents_no, dim);
y = arrayfun(@(i) fobj(x(i,:)), 1:SearchAgents_no)';
p_y = y;
x_hist_best = x;
g_hist_best = x;
y_hist_best = inf(SearchAgents_no, 1);
x_global_best = x(1, :);
g_global_best = zeros(1, dim);
y_global_best = fobj(x_global_best);
hop = ub - lb;
Convergence_curve = zeros(Max_iter, 1);

% --- Main Loop ---
tic;
for t = 1:Max_iter
    for i = 1:SearchAgents_no
        % --- Gradient Estimation ---
        p_y(i) = sum(w(i,:) .* x(i,:));
        p = p_y(i) - y(i);
        g_temp = p .* x(i,:);

        % Individual Direction
        p_d = (x_hist_best(i,:) - x(i,:)) .* (y_hist_best(i) - y(i));
        p_d = p_d ./ ((sum(p_d) + eps)^2);
        d_p = p_d + g_hist_best(i,:);

        % Group Direction
        c_d = (x_global_best - x(i,:)) .* (y_global_best - y(i));
        c_d = c_d ./ ((sum(c_d) + eps)^2);
        d_g = c_d + g_global_best;

        % Final Gradient
        r1 = rand(1, dim); r2 = rand(1, dim);
        g = (1 - r1 - r2).*g_temp + r1.*d_p + r2.*d_g;
        g = g ./ (sum(g) + eps);

        % Momentum Update
        m(i,:) = 0.9 * m(i,:) + 0.1 * g;
        v(i,:) = 0.99 * v(i,:) + 0.01 * g.^2;
        w(i,:) = w(i,:) - m(i,:) ./ (sqrt(v(i,:)) + eps);

        % --- Advice Forward ---
        x_o = x(i,:) + exp(-t/(0.1*Max_iter)) * 0.1 .* hop .* g;
        x_o = min(max(x_o, lb), ub);
        y_o = fobj(x_o);

        % --- Random Search ---
        r = rand(1, dim) * pi - pi/2;
        x_n = x(i,:) + tan(r) .* hop / (1 + t) * 0.5;
        x_n = min(max(x_n, lb), ub);
        y_n = fobj(x_n);

        % --- Encircling Mechanism ---
        d = x_hist_best(i,:) - x(i,:);
        d_g = x_global_best - x(i,:);
        r1 = rand(1, dim); r2 = rand(1, dim);
        x_m = (1 - r1 - r2).*x(i,:) + r1.*d + r2.*d_g;
        x_m = min(max(x_m, lb), ub);
        y_m = fobj(x_m);

        % --- Discriminant Selection ---
        x_summary = [x_m; x_n; x_o];
        y_summary = [y_m, y_n, y_o];
        y_summary(isnan(y_summary)) = inf;
        [y_i, idx] = min(y_summary);
        x_i = x_summary(idx, :);

        if y_i < y(i)
            y(i) = y_i;
            x(i,:) = x_i;
            if y_i < y_hist_best(i)
                y_hist_best(i) = y_i;
                x_hist_best(i,:) = x_i;
                g_hist_best(i,:) = g_temp;
                if y_i < y_global_best
                    y_global_best = y_i;
                    x_global_best = x_i;
                    g_global_best = g_temp;
                end
            end
        elseif rand() < 0.3
            y(i) = y_i;
            x(i,:) = x_i;
        end
    end

    % --- Logging and Convergence ---
    fprintf('%d, %f\n', t, y_global_best);
    Convergence_curve(t) = y_global_best;
end
runtime = toc;
end
