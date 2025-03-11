function real_coord(Shull,path)
    %UNTITLED2 Summary of this function goes here
    %   Detailed explanation goes here
    all_coords = double.empty(0,4);

    for idx = 1:1:length(Shull.real_coord)
    if sum(iscell(Shull.real_coord{idx})) == 0
        real_cord = [999 999 999 Shull.frames(idx)];
    else
        add_dim = max(cellfun(@length, Shull.real_coord{idx}));
        real_cord = nan(add_dim,4);
        for xyz = 1:1:3
            real_cord(1:length(Shull.real_coord{idx}{xyz}),xyz) = Shull.real_coord{idx}{xyz}';
        end
    end
    real_cord(:,4) = Shull.frames(idx);
    all_coords = [all_coords;real_cord];
    end

    save(path,'all_coords');

end
