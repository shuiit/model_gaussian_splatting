clear; close all; clc;

%% Settings
num_cams  = 4;
frame     = 1341;
base_dir  = 'G:\My Drive\Research\gaussian_splatting\gaussian_splatting_input\mov94_2023_08_09_60ms\images\';
frames_dir= 'G:\My Drive\Research\gaussian_splatting\article\plots\3DV\frames\';

% --- define desired half-sizes (in pixels) relative to CM ---
deltaH = 60;   % half-height
deltaW = 60;   % half-width
desired_height = 2*deltaH + 1;   % final crop height
desired_width  = 2*deltaW + 1;   % final crop width

row_centers = zeros(num_cams,1);   % CM row
col_centers = zeros(num_cams,1);   % CM col

%% Pass 1: compute CM (center-of-mass) per image
for cam_num = 1:num_cams
    image_name = sprintf('P%dCAM%d.mat', frame, cam_num);
    S = load([base_dir, image_name]);

    mask = S.im > 0;
    [rows, cols] = find(mask);
    % center-of-mass of mask (in image coords)
    row_centers(cam_num) = mean(rows);
    col_centers(cam_num) = mean(cols);

    % (Optional) visualize mask:
    % figure; imshow(mask, []); title(sprintf('Mask CAM%d', cam_num));
end

%% Pass 2: crop each image to (desired_height, desired_width) centered at CM
for cam_num = 1:num_cams
    png_out   = sprintf('P%dCAM%d.png', frame, cam_num);
    image_mat = sprintf('P%dCAM%d.mat', frame, cam_num);

    % Load image + per-cam background
    S  = load([base_dir, image_mat]);
    bg = load([base_dir, sprintf('cam%d_bg.mat', cam_num)]);

    img = S.im;
    image_to_plot = img + im2double(bg.bg) .* (1 - 1*(img > 0));

    [H, W] = size(image_to_plot);

    % CM for this image
    rc = row_centers(cam_num);
    cc = col_centers(cam_num);

    % Desired crop indices (centered on CM)
    r1 = round(rc) - deltaH;
    r2 = r1 + desired_height - 1;
    c1 = round(cc) - deltaW;
    c2 = c1 + desired_width  - 1;

    % Padding if the window goes out of bounds
    pad_top    = max(1 - r1, 0);
    pad_left   = max(1 - c1, 0);
    pad_bottom = max(r2 - H, 0);
    pad_right  = max(c2 - W, 0);

    if any([pad_top pad_left pad_bottom pad_right] > 0)
        pad_val = 0; % choose background tone if desired
        X = padarray(image_to_plot, [pad_top pad_left],   pad_val, 'pre');
        X = padarray(X,             [pad_bottom pad_right], pad_val, 'post');

        % Shift indices into padded image
        r1 = r1 + pad_top; r2 = r2 + pad_top;
        c1 = c1 + pad_left; c2 = c2 + pad_left;

        cropped = X(r1:r2, c1:c2);
    else
        cropped = image_to_plot(r1:r2, c1:c2);
    end

    % Enforce exact size (guards off-by-one at borders)
    if ~isequal(size(cropped), [desired_height, desired_width])
        cropped = imresize(cropped, [desired_height, desired_width], 'nearest');
    end

    imwrite(cropped, [frames_dir, png_out]);
    % (Optional) preview
    figure; imshow(cropped, []); 
end
