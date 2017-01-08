% Number of images in the data set.
num_img = 10;

% Will keep the count of total number of objects.
num_obj = 0;

% Step value for dense sampling.
step_ = 6;
% Size value for dense sampling.
size_ = 8;

% Number of clusters for kmeans.
num_clusters = 10;

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
end

% Simple function to draw points on whence the descriptors are obtained.
% dsift_img = get_dsift_labels(images{1}, frames);
% imshow(dsift_img);

% Put all the descriptors in a single matrix.
all_descriptors = zeros(128, total_descriptors);
count = 0;
for i = 1:num_img
    tmp = size(descriptors{i},2);
    for j = 1:tmp
        all_descriptors(:,count + j) = descriptors{i}(:,j);
    end
    count = count + tmp;
end

% Apply kmeans. Centers is the centroid of each cluster.
% Labels is the respective cluster each descriptor belongs to.
[centers, labels] = vl_kmeans(all_descriptors, num_clusters);

% @TODO change the following code to have histograms for each individual
% object.
histograms = cell(num_img, 1);
% Create a histogram for each image using the kmeans results.
for i = 1:num_img
    % Start and end indices for the current image.
    start_index = sum(descriptor_counts(1:i-1));
    if start_index == 0
        start_index = 1;
    end
    end_index = start_index + descriptor_counts(i);
    [histogram, edges] = histcounts(labels(start_index:end_index));
    histograms{i} = histogram;
end