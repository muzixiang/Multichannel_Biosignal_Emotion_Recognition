
trialNum = 40;
trialTime = 60;
augmentNum = 250;%How many times to expand the data through add noise.
window = 2;%sliding time window
seqL = trialTime/window;%how many elements a sequence contains
channelNum = 32;
scaleNum = 45;

totalTrialNum = trialNum*augmentNum;

%估计可能得到的总sequence数
seqNum = totalTrialNum;

for subNo = 1:32
    X = zeros(seqNum,seqL,1,channelNum,scaleNum);
    Y_general = zeros(seqNum,3);
    Y_personal = zeros(seqNum,3);
    
    filepath1 = strcat('D:\LX\element2ds\trial_cwt\',num2str(window),'s_as_element\element2ds_sub',num2str(subNo));
    filepath2 = 'C:\Users\LX\我的坚果云\Project Files\Matlab\DEAP2\label\trial_labels_general_valence_arousal_dominance.mat';%general label
    filepath3 = 'C:\Users\LX\我的坚果云\Project Files\Matlab\DEAP2\label\trial_labels_personal_valence_arousal_dominance.mat';%subjective label
    data1 = load(filepath1);
    element2ds = data1.element2ds;
    data2 = load(filepath2);
    data3 = load(filepath3);
    general_trial_labels = data2.trial_labels;
    personal_trial_labels = data3.trial_labels;
    general_sub_trial_labels = general_trial_labels((subNo-1)*trialNum+1:subNo*trialNum,:);
    personal_sub_trial_labels = personal_trial_labels((subNo-1)*trialNum+1:subNo*trialNum,:);
    
    %Construct X from element2ds
    seqNo = 1;
    for trialNo = 1:totalTrialNum
        element2ds_trial = element2ds(trialNo,:,:,:);
        element2ds_trial = squeeze(element2ds_trial);
        squence = element2ds_trial;
        X(seqNo,1:seqL,1,:,:) = squence; %store the selected sequences into X
        label_index = ceil(trialNo/augmentNum);
        Y_general(seqNo,:) = general_sub_trial_labels(label_index);%Store the sequence's corresponding label in the Y
        Y_personal(seqNo,:) = personal_sub_trial_labels(label_index);
        fprintf(strcat('sub-',num2str(subNo),' seqL-',num2str(seqL),' trial-',num2str(trialNo),' seqNo-',num2str(seqNo),'\n'));
        seqNo = seqNo+1;%update the write index
    end
    
    %store the labels of the 40 trials simultaneously for further usage
    Y_40general(:,:) = general_sub_trial_labels(:,:);
    Y_40personal(:,:) = personal_sub_trial_labels(:,:);
        
    %save the constructed X and Y
    filename = strcat('D:\LX\RCNNXY\trial_cwt\nooverlap\',num2str(window),'s_as_element\sub',num2str(subNo),'.mat');
    save(filename,'X','Y_general','Y_personal','Y_40general','Y_40personal','-v7.3');
end