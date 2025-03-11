function plot_camera(rotation,translation,rotate,ttl)
clr = {'r','g','b','m'}
for j = 1:1:4
    trans = rotate*translation(:,:,j);
    rot = rotate*rotation(:,:,j)';
    
    for k = 1:1:3
        quiverHandles(k) = quiver3(trans(1),trans(2),trans(3),rot(1,k),rot(2,k),rot(3,k),0.1,color = clr{k});hold on
    end
    scatterHandles(j) = scatter3(trans(1),trans(2),trans(3),clr{j},'filled');hold on
end
scatter3(0,0,0,100,'filled')
legend([quiverHandles,scatterHandles], {'x', 'y', 'z','cam1','cam2','cam3','cam4'});
xlabel('x');ylabel('y');zlabel('z')
title(ttl);axis equal
end