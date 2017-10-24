close all;
clear;

% -- settings start here ---
% set 1 to use gpu, and 0 to use cpu
use_gpu = 0;
bit_48=zeros(5,3);
for iii=1:5
    % top K returned images
    top_k = iii*1000;
    feat_len =24;  % binary code length
    caffe.reset_all();      
    % set result folder
    result_folder = './analysis';

    % models
    model_file = '/home/leijzhan/my_caffe/Workspace/cifar-10/ssdh_tanh24/train_iter_80000.caffemodel';
    % model definition
    model_def_file = '/home/leijzhan/my_caffe/Workspace/cifar-10/ssdh_tanh24/deploy_tanh24.prototxt';



    % % models
    % model_file = './101/tanh128/train_iter_120000.caffemodel';
    % % model definition
    % model_def_file = './models/101/deploy_101_tanh128.prototxt';




    % test_file_list = './lmdb/test-file-list.txt';
    % test_label_file = './lmdb/test-label.txt';
    % train_file_list = './lmdb/train-file-list.txt';
    % train_label_file = './lmdb/train-label.txt';
    % 
    test_file_list = './data/cifar10/test-file-list.txt';
    test_label_file = './data/cifar10/test-label.txt';
    train_file_list = './data/cifar10/train-file-list.txt';
    train_label_file = './data/cifar10/train-label.txt';
    % caffe mode settingbinary_test_file
    phase = 'test'; % run with phase test (so that dropout isn't applied)


    % --- settings end here ---

    % outputs
    feat_test_file = sprintf('%s/feat-test_t_%dbit.mat', result_folder,feat_len);
    feat_train_file = sprintf('%s/feat-train_t_%dbit.mat', result_folder,feat_len);
    binary_test_file = sprintf('%s/binary-test_t_%dbit.mat', result_folder,feat_len);
    binary_train_file = sprintf('%s/binary-train_t_%dbit.mat', result_folder,feat_len);

    
    
        % feature extraction- test set
    if exist(binary_test_file, 'file') ~= 0
        load(binary_test_file);
    else
        
        feat_test = feat_batch(use_gpu, model_def_file, model_file, test_file_list, feat_len);
        save(feat_test_file, 'feat_test', '-v7.3');
        
        binary_test = (feat_test>0);
        save(binary_test_file,'binary_test','-v7.3');
    end
    
    % feature extraction- training set
    if exist(binary_train_file, 'file') ~= 0
        load(binary_train_file);
    else
        
        feat_train = feat_batch(use_gpu, model_def_file, model_file, train_file_list, feat_len);
        save(feat_train_file, 'feat_train', '-v7.3');
        binary_train = (feat_train>0);
        
        save(binary_train_file,'binary_train','-v7.3');
    end

    trn_label = load(train_label_file);
    tst_label = load(test_label_file);

    [map, precision_at_k,recall_at_k] = precision( trn_label, binary_train, tst_label, binary_test, top_k, 1);
    bit_48(iii,1)=map;bit_48(iii,2)=mean(precision_at_k);bit_48(iii,3)=recall_at_k;
    fprintf('MAP = %f\n',map);
    % map and precision outputs
%     map_file = sprintf('%s/map%dbit%dsmpl.mat', result_folder,feat_len,iii*1000);
%     save(map_file, 'map', '-ascii');
    P = [[1:1:top_k]' precision_at_k'];
    precision_file = sprintf('%s/precision_t_%dbit%dsmpl.mat', result_folder,feat_len,iii*1000);
    save(precision_file, 'P', '-ascii');
    save_path = sprintf('%s/lei_t_%dbit.mat', result_folder,feat_len);
    save (save_path,'bit_48');
end



