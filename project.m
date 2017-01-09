% Number of images in the data set.
num_img = 20;

% Will keep the count of total number of objects.
num_obj = 0;
train_num_obj = 0;
% Number of object types.
num_obj_type = 8;

% Step value for dense sampling.
step_ = 10;
% Size value for dense sampling.
size_ = 8;

% Number of clusters for kmeans.
num_clusters = 500;

% Images in the data set.
images = cell(num_img,1);
% Masks in the data set.
masks = cell(num_img,1);
frames = cell(num_img,1);

% Descriptors for each image.
descriptors = cell(num_img,1);
% Number of descriptors for each image.
descriptor_counts = zeros(num_img, 1);
% Number of objects for each image.
object_counts = zeros(num_img, 1);
% Total number of descriptors.
total_descriptors = 0;

object_types = {'screen', 'keyboard', 'mouse', 'mug', 'car', 'tree', 'person', 'building'};
% 1: computer screen
% 2: keyboard
% 3: mouse
% 4: mug
% 5: car
% 6: tree
% 7: person
% 8: building
object_total = [90, 88, 77, 43, 53, 68, 33, 92];
train_object_total = zeros(num_obj_type, 1);

train_ratio = 0.5;
test_ratio = 0.5;
valid_ratio = 0;
[train_ind, valid_ind, test_ind] = dividerand(num_img, train_ratio, valid_ratio, test_ratio);


% Read the images and the masks for each image.
% Create descriptors for each image using dense sampling.
for i = 1:num_img
    % Read the masks.
    masks{i} = load(strcat('data/', num2str(i), '.mat'));
    % Read the images.
    images{i} = imread(strcat('data/', num2str(i), '.jpg'));
    % Convert image to grayscale before applying DSIFT.
    tmp_im = single(rgb2gray(images{i}));
    % Apply SIFT using dense sampling with parameters defined above.
    [tmp_f, tmp_d] = vl_dsift(tmp_im, 'Step', step_, 'Size', size_);
    
    frames{i} = tmp_f;
    descriptors{i} = tmp_d;
    descriptor_counts(i) = length(tmp_d);
    total_descriptors = total_descriptors + length(tmp_d);
    
    object_counts(i) = length(masks{i}.masks);
    num_obj = num_obj + object_counts(i);
    if ~isempty(find(train_ind == i,1))
        cur = masks{i}.masks;
        for j = 1:length(cur)
            cur_object_name = cur(j).class_name;
            cur_index = 0;
            % Find the index of the current object's type.
            for k = 1:num_obj_type
                if strcmp(cur_object_name, object_types{k})
                    cur_index = k;
                    break
                end
            end
            train_object_total(cur_index) = train_object_total(cur_index) + 1;
            train_num_obj = train_num_obj + 1;
        end
    end
end



% Simple function to draw points on whence the descriptors are obtained.
% dsift_img = get_dsift_labels(images{1}, frames);
% imshow(dsift_img);

% Put all the descriptors in a single matrix for clustering.
all_descriptors = zeros(128, total_descriptors);
count = 0;
for i = 1:num_img
    tmp = size(descriptors{i},2);
    for j = 1:tmp
        all_descriptors(:,count + j) = descriptors{i}(:,j);
    end
    count = count + tmp;
end

% Apply kmeans clustering. centers is a vector of coordinates of the centroid of each cluster.
% labels is the respective cluster each descriptor belongs to.
[centers, labels] = vl_kmeans(all_descriptors, num_clusters);

% Histograms for each object in the training set.
object_histograms = zeros(train_num_obj, num_clusters);

% Histograms for the objects of each object type in the training set.
% Follows the indexing scheme from above.
histograms = cell(num_obj_type, 1);
for i = 1:num_obj_type
    histograms{i} = zeros(train_object_total(i), num_clusters);
end

% Counts of how many objects of certain types have been found.
cur_counts = zeros(num_obj_type, 1);

% Count of which object this is.
cur_obj_count = 0;

% Iterate through each image in the training set.
for c_i = 1:length(train_ind)
    i = train_ind(c_i);
    % Mask and frame of the current image.
    cur_mask = masks{i};
    cur_frame = frames{i};
    % Iterate through each object in the current image.
    for j = 1:object_counts(i,1)
        % Mask for the current ojbect.
        cur_object_mask = cur_mask.masks(j).mask;
        % Current object's class name
        cur_object_name = cur_mask.masks(j).class_name;
        
        cur_index = 0;
        % Find the index of the current object's type.
        for k = 1:num_obj_type
            if strcmp(cur_object_name, object_types{k})
                cur_index = k;
                break
            end
        end
        
        cur_counts(cur_index) = cur_counts(cur_index) + 1;
        cur_obj_count = cur_obj_count + 1;
        
        % Base index for the first descriptor in this image.
        base_index = sum(descriptor_counts(1:i-1));
        
        % Iterate through this image's descriptors.
        k = 1;
        while k < descriptor_counts(i)
            x = cur_frame(k);
            y = cur_frame(k+1);
            if cur_object_mask(x,y) == 1
                cur_label = labels(base_index + k);
                temp = object_histograms(cur_obj_count, cur_label);
                % Increment the count of the corresponding bin in the
                % histogram.
                object_histograms(cur_obj_count, cur_label) = temp + 1;
                
                % Find the count of the current object type so far;
                cc = cur_counts(cur_index);
                histograms{cur_index}(cc, cur_label) = histograms{cur_index}(cc, cur_label) + 1;
            end
            
            % Increment k by 2.
            k = k + 2;
        end  
    end
end


