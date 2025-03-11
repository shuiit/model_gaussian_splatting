function [] = hull_mat_file(data,path,frames)

hull = uint16.empty(0,4);



for idx = 1:1:length(frames)
    if iscell(data{idx}) || length(data{idx}) ==3
        hull = [hull;[999,999,999,frames(idx)]];
    else
        hull = [hull;[data{idx},frames(idx)*ones(length(data{idx}),1)] ];
    end

end

save(path,'hull')

end