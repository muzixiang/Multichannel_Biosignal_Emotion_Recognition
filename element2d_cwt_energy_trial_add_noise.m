wname = 'db4';
scaleNum = 45;
scales = 1:1:scaleNum;

trialTime = 60;
Fs = 128;
window = 2;%4s的信号作为一个element
eleL = window*Fs;%一步一个element包含多少采样点，这个地方可以修改，比如0.5Fs,1Fs,1.5Fs，2Fs,3Fs,4Fs,5Fs,6Fs
eleNum = (Fs*trialTime)/eleL;%理解成多少个时间step，时间窗。每个时间窗内的scalogram数据进行合并成一个向量。

augmentNum = 250;%增加噪声数据扩张250倍
channelNum=32;
trialNum=40;

element2ds = zeros(trialNum*augmentNum,eleNum,channelNum,scaleNum);

for subNo=1:32
   if subNo<10
        filePath = strcat('D:\DEAP\matlab format data\s0',num2str(subNo),'.mat');
    else
        filePath = strcat('D:\DEAP\matlab format data\s',num2str(subNo),'.mat');
    end
    datFile = load(filePath);
    eegdata = datFile.data;
    for trialNo=1:40
        for channelNo=1:32
            trialCount = (trialNo-1)*augmentNum+1;%计数器，用来记录当前处理的是哪个trial的数据，因为进行了augment,所以需要除trialNo外的另外一个计数器。
            channelSignal = eegdata(trialNo,channelNo,:);
            channelSignal = squeeze(channelSignal(1,1,:));
            channelTrialSignal = channelSignal(128*3+1:end);
            coefs = cwt(channelTrialSignal,scales,wname);
            scalogram = wscalogram('[]',coefs,'scales',scales,'ydata',channelTrialSignal);
            for ele = 1:eleNum
                scalogram_t = scalogram(:,(ele-1)*eleL+1:ele*eleL);
                scalsum_t = sum(scalogram_t,2);
                scalsum_t = scalsum_t./100;
                element2ds(trialCount,ele,channelNo,:)=scalsum_t;
                fprintf('cwt processing subject %d realtrial %d augtrial %d channel %d element %d \n ', subNo, trialNo, trialCount, channelNo, ele);
            end
            trialCount = trialCount+1;
            for aug=1:(augmentNum-1) %对当前的channelTrialSignal进行增加噪声数据扩倍
                noisedSig = awgn(channelTrialSignal,5,'measured');
                coefs = cwt(noisedSig,scales,wname);
                scalogram = wscalogram('[]',coefs,'scales',scales,'ydata',noisedSig);
                for ele = 1:eleNum
                    scalogram_t = scalogram(:,(ele-1)*eleL+1:ele*eleL);
                    scalsum_t = sum(scalogram_t,2);
                    scalsum_t = scalsum_t./100;
                    element2ds(trialCount,ele,channelNo,:)=scalsum_t;
                    fprintf('cwt processing subject %d realtrial %d channel %d augtrial %d  augment %d element %d \n ', subNo, trialNo, channelNo, trialCount, aug, ele);
                end
                trialCount = trialCount+1;
            end
        end
    end
    
    filename = strcat('D:\LX\element2ds\trial_cwt\',num2str(window),'s_as_element\element2ds_sub',num2str(subNo),'.mat');
    save(filename,'element2ds','-v7.3');
end