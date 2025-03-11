function [Q, R] = QR_Decomposition(A)
    A = [A(:,1),A(:,2:3)];
    [n, m] = size(A);  % Get the dimensions of A

    Q = zeros(n, n);   % Initialize matrix Q
    u = zeros(n, n);   % Initialize matrix u

    u(:, 1) = A(:, 1);
    Q(:, 1) = u(:, 1) / norm(u(:, 1));  % Compute the first column of Q

    for i = 2:n
        u(:, i) = A(:, i);
        for j = 1:(i-1)
            u(:, i) = u(:, i) - (A(:, i)' * Q(:, j)) * Q(:, j);  % Get each u vector
        end
        Q(:, i) = u(:, i) / norm(u(:, i));  % Compute each e vector
    end
    Q = [Q(:,1:2),Q(:,3)];
    A = [A(:,1:2),A(:,3)];
    R = zeros(n, m);   % Initialize matrix R
    for i = 1:n
        for j = i:m
            R(i, j) = Q(:, i)' * A(:, j);  % Compute the R matrix entries
        end
    end
end