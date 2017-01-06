num_img = 188;

images = cell(num_img,1);
masks = cell(num_img,1);
frames = cell(num_img,1);
descriptors = cell(num_img,1);
total_descriptors = 0;
for i = 1:num_img
    masks{i} = load(strcat('data/', num2str(i), '.mat'));
    images{i} = imread(strcat('data/', num2str(i), '.jpg'));
    tmp_im = single(rgb2gray(images{i}));
    [tmp_f, tmp_d] = vl_dsift(tmp_im, 'Step', 12, 'Size', 8);
    frames{i} = tmp_f;
    descriptors{i} = tmp_d;
    total_descriptors = total_descriptors + length(tmp_d);
end

copy_img = images{1};
for i = 1:size(frames{1}, 2)
    [cur_x, cur_y] = frames{1}(i,:);
end
% all_descriptors = zeros(128, total_descriptors);
% count = 0;
% for i = 1:num_img
%     tmp = size(descriptors{i},2);
%     for j = 1:tmp
%         all_descriptors(:,count + j) = descriptors{i}(:,j);
%     end
%     count = count + tmp;
% end
% [c, a] = vl_kmeans(all_descriptors, 1500);
