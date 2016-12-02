wname='db4';
scaleNum = 64;
subNum = 32;
trialNum = 40;
channelNum = 32;
sub_avg_energy_entropy_ratio = zeros(scaleNum,subNum);
for subNo = 1:subNum
    energy_entropy_ratios = zeros(scaleNum,trialNum*channelNum);
    if subNo<10
        filePath = strcat('D:\DEAP\matlab format data\s0',num2str(subNo),'.mat');
    else
        filePath = strcat('D:\DEAP\matlab format data\s',num2str(subNo),'.mat');
    end
    datFile = load(filePath);
    eegdata = datFile.data;
    index = 1;
    for trialNo =1:trialNum
        for channelNo = 1:channelNum
            channelSignal = eegdata(trialNo,channelNo,:);
            trialSignal = channelSignal(128*3+1:end);
            scales=1:1:scaleNum;
            coefs = cwt(trialSignal,scales,wname);
            coefs_Energy = coefs.^2;
            scale_Energy = sum(coefs_Energy,2);%计算各个尺度的能量 calculate the energy in each scale
            sizecoefs = size(coefs,2);
            scale_Energy = repmat(scale_Energy,1,sizecoefs);
            Pi = coefs_Energy./scale_Energy;%计算获得各个系数能量与相应尺度能量的比值
            logPi = log(Pi);
            scale_Entropy = -sum(Pi.*logPi,2);%计算各个尺度的熵值 calculate the entropy of each scale
            energy_entropy_ratio = scale_Energy(:,1)./scale_Entropy;
            energy_entropy_ratios(:,index) = energy_entropy_ratio;
            index = index+1;
            fprintf('sub %d trial %d channel %d \n', subNo, trialNo, channelNo);
        end
    end
    avg_energy_entropy_ratio = mean(energy_entropy_ratios,2);
    sub_avg_energy_entropy_ratio(:,subNo) = avg_energy_entropy_ratio;
end
for subNo = 1:subNum
    subplot(1,4,subNo);
    plot(1:1:scaleNum, sub_avg_energy_entropy_ratio(:,subNo),'*');
    xlabel(strcat(num2str(subNo),'Scales'));
    ylabel('Energy/Entropy');
end
