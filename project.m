% Number of images in the data set.
num_img = 188;

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
% Better masks for future usage, putting together separate instance of the same object type.
better_masks = cell(num_img, num_obj_type);

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
% Total number of objects by type in the training set.
train_object_total = zeros(num_obj_type, 1);

% Ratios of training and test set, to be divided randomly.
train_ratio = 0.5;
test_ratio = 0.5;
valid_ratio = 0;
[train_ind, valid_ind, test_ind] = dividerand(num_img, train_ratio, valid_ratio, test_ratio);

% Read the images and the masks for each image.
% Create descriptors for each image using dense sampling.
for i = 1:num_img
    fprintf('Loop %d iteration %d\n', 1, i);
    % Read the masks.
    masks{i} = load(strcat('data/', num2str(i), '.mat'));
    % Read the images.
    images{i} = imread(strcat('data/', num2str(i), '.jpg'));
    
    % Initialize the indices of the better masks to zero.
    % They are to be orred with the given masks.
    for j = 1:num_obj_type
        better_masks{i,j} = false(size(images{i}, 1), size(images{i}, 2));
    end
    
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
        b = better_masks{i, cur_index};
        a = cur(j).mask;
        better_masks{i, cur_index} = (a|b);
        if ~isempty(find(train_ind == i,1))
            train_object_total(cur_index) = train_object_total(cur_index) + 1;
            train_num_obj = train_num_obj + 1;
        end
    end
end
% 
% Training data labels for each object type.
% Binary labels for each object.
train_labels = cell(num_obj_type, 1);
for i = 1:num_obj_type
    train_labels{i} = zeros(train_num_obj, 1);
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
    fprintf('Loop %d iteration %d\n', 2, c_i);
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
        
        for t = 1:num_obj_type
            train_labels{t}(cur_obj_count) = -1;
        end
        % Mark the corresponding class name's label as 1 for the current
        % object.
        train_labels{cur_index}(cur_obj_count) = 1;
        
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

models = cell(num_obj_type, 1);
% Train binary SVM classifiers for each object type.
for i = 1:num_obj_type
    cur_model = fitcsvm(object_histograms, train_labels{i}, 'BoxConstraint', 1, 'KernelFunction', 'polynomial');
    models{i} = cur_model;
end

% Load pre-computed segmentations of images.
segments = load('segments.mat');
segments = segments.segments;
% Load the number of segments for each image.
numcuts = load('numcuts.mat');
numcuts = numcuts.num_cuts;

% Number of test objects, init to 0.
test_num_obj = 0;
for i = 1:length(test_ind)
    test_num_obj = test_num_obj + numcuts(1, test_ind(i));
end

% Test histograms
test_histograms = zeros(test_num_obj, num_clusters);
% Counter for object index.
cur_object = 0;

for c_i = 1:length(test_ind)
    fprintf('Loop %3 iteration %d\n', 3, c_i);
    % Get the current image from the test set.
    i = test_ind(c_i);
    % Perform segmentations for the images in the test data.
    % [SegLabel,NcutDiscrete,NcutEigenvectors,NcutEigenvalues,W,imageEdges]= NcutImage(images{c_i},length(masks{i}.masks));
    
    % Indices of the descriptors for the current image.
    cur_frame = frames{i};
    
    % Base index for the first descriptor in this image.
    base_index = sum(descriptor_counts(1:i-1));
    
    % Iterate through this image's descriptors.
    k = 1;
    while k < descriptor_counts(i)
        x = cur_frame(k);
        y = cur_frame(k+1);
        
        cur_label = labels(base_index + k);
        
        % Find the index for the current object.
        cur_object_label = cur_object + segments{i}(x, y);
        
        % Increment the count of the current bin for the test.
        test_histograms(cur_object_label, cur_label) = test_histograms(cur_object_label, cur_label) + 1;
        
        k = k + 2;
    end
    
    % Add the number of objects processed in this iteration to the counter.
    cur_object = cur_object + numcuts(1,i);
end

% Allocate cell arrays for predictions and scores.
predictions = cell(num_obj_type, 1);
scores = cell(num_obj_type, 1);
for i = 1:num_obj_type
    predictions{i} = zeros(test_num_obj, 1);
    scores{i} = zeros(test_num_obj, 1);
end

% Predict the labels for the given test objects.
for i = 1:num_obj_type
    [label, score] = predict(models{i}, test_histograms);
    predictions{i} = label;
    scores{i} = score;
end

% Compute the probability maps for each segment.
max_prob = zeros(test_num_obj, 1);
% Object type from where the maximum score comes.
type_obj = zeros(test_num_obj, 1);

for i = 1:test_num_obj
    fprintf('Loop %4 iteration %d\n', 4, i);
    prob_max = 0;
    prob_max_ind = 0;
%     sum_prob = 0;
    for j = 1:num_obj_type
        cur_l = predictions{j}(i);
        if (cur_l == 1)
            % The probability of it being the one of the current object type.
            cur_prob = abs(scores{j}(i));
            % Save the current maximum probability and its index.
            if cur_prob > prob_max
                prob_max = cur_prob;
                prob_max_ind = j;
            end
        end
        
%         sum_prob = sum_prob + cur_prob;
    end
    
    % Update the maximum confidence and the object type that it belongs to.
    max_prob(i) = prob_max;
    type_obj(i) = prob_max_ind;
end

% Probability maps for all test images.
prob_maps = cell(length(test_ind), 1);
% The types of objects that each segment is classified as.
prob_maps_inds = cell(length(test_ind), 1);

segment_count = 0;

for c_i = 1:length(test_ind)
    fprintf('Loop %d iteration %d\n', 5, c_i);
    i = test_ind(c_i);
    img = images{i};
    
    % Initialize the probability maps to the same size as the images.
    prob_maps{c_i} = zeros(size(images{i}, 1), size(images{i}, 2));
    prob_maps_inds{c_i} = zeros(size(images{i}, 1), size(images{i}, 2));
    
    segment = segments{i};
    num_cut = numcuts(1, i);
    
    % Add the predictions and the scores to the map for each segment in the
    % current image.
    for j = 1:num_cut
        prob_maps{c_i}(segment == j) = max_prob(segment_count + j);
        prob_maps_inds{c_i}(segment == j) = type_obj(segment_count + j);
    end
    
    segment_count = segment_count + num_cut;
end

% Accuracies. [test_index, object_type, (tp,fp,fn,tn)]
accuracies = cell(length(test_ind), 1);
for i = 1:length(test_ind)
    accuracies{i} = zeros(num_obj_type, 4);
end
% A threshold to be trialed.
threshold = 50;

% Iterate through the test images.
for c_i = 1:length(test_ind)
    fprintf('Loop %d iteration %d\n', 6, c_i);
    % Get the current test index.
    i = test_ind(c_i);
    img = images{i};
    
    % Iterate through object types.
    % The accuracies for each one will be calculated separately.
    for j = 1:num_obj_type
        % Get the mask for the current image, with each instance of the
        % current object type marked as 1.
        better_mask = better_masks{i, j};
        current_mask = (prob_maps_inds{c_i} == j & prob_maps{c_i} > threshold);
        
        % True positive:
        % Exists in given mask and the found mask.
        tp = sum(sum(better_mask & current_mask));
        % False positive:
        % Not in the given mask but in the found mask.
        fp = sum(sum(~better_mask & current_mask));
        % False negative:
        % In the given mask but not in the found mask.
        fn = sum(sum(better_mask & ~current_mask));
        % True negative:
        % Exists in neither masks.
        tn = sum(sum(~better_mask & ~current_mask));
        
        accuracies{c_i}(j, 1) = tp;
        accuracies{c_i}(j, 2) = fp;
        accuracies{c_i}(j, 3) = fn;
        accuracies{c_i}(j, 4) = tn;
    end
end

