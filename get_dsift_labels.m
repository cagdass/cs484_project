function dsift_img = get_dsift_labels(img, frames)
dsift_img = img;
for i = 1:length(frames{1})
    a = frames{1}(:,i);
    cur_x = a(1);
    cur_y = a(2);
    for x = cur_x-1:cur_x+1
        for y = cur_y-1:cur_y+1
            if y > 0 && y < size(dsift_img,2) && x > 0 && x < size(dsift_img,1)
                dsift_img(x,y,1) = 255;
                dsift_img(x,y,2) = 200;
                dsift_img(x,y,3) = 100;
            end
        end
    end
end