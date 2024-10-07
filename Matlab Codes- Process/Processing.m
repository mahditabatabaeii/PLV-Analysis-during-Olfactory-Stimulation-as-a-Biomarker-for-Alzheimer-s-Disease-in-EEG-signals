%% PLV and Phase Differenece and Heatmap Matrix
clc 
clear

% Load the 'normal' and 'AD' datasets
normal = load('Dataset\Normal.mat').normal;
AD = load('Dataset\AD.mat').AD;
MCI = load('Dataset\MCI.mat').MCI;

% Set the frequency range and sampling frequency
freqRange = [35 40];
fs = 200;

% Set the number of channels, and the indices of the two channels of interest
numChannels = 4;
channel1 = 2;
channel2 = 3;

% Set the number of individuals in the 'normal' and 'AD' datasets
normalIndividuals = 15;
ADIndividuals = 13;
MCIindividuals = 7;

% Calculate PLV for the 'normal' and 'AD' dataset
[plvNormalFreq, plvNormalRare] = plv2Channels(normal, fs, freqRange, normalIndividuals, channel1, channel2);
[plvADFreq, plvADRare] = plv2Channels(AD, fs, freqRange, ADIndividuals, channel1, channel2);

% Phase Difference - Random Subject
[phaseDifferenceNormalFreq, phaseDiffereneceNormalRare] = avgPhase2Channels(normal, channel1,channel2, 9); % for 9th person
[phaseDifferenceADFreq, phaseDiffereneceADRare] = avgPhase2Channels(AD, channel1,channel2, 7); % for 7th person

% Phase Difference - Mean
[meanPhaseDifferenceNormalFreq, meanPhaseDiffereneceNormalRare] = meanPahseDiff(normal, channel1,channel2, normalIndividuals);
[meanPhaseDifferenceADFreq, meanPhaseDiffereneceADRare] = meanPahseDiff(AD, channel1,channel2, ADIndividuals);
 
% HeatMap Matrix
plvNormalFreqMatrix = zeros();
plvNormalRareMatrix = zeros();
plvADFreqMAtrix = zeros();
plvADRareMatrix = zeros();
plvMCIFreqMatrix = zeros();
plvMCIRareMatrix = zeros();

% P-Value in Mtrix
pValFreq = zeros();
pValRare = zeros();

for i = 1:numChannels
    for j = 1:numChannels
        [temp1, temp2] = plv2Channels(normal, fs, freqRange, normalIndividuals, i, j);

        plvNormalFreqMatrix(i,j) = mean(temp1);
        plvNormalRareMatrix(i,j) = mean(temp2);

        [temp3, temp4] = plv2Channels(AD, fs, freqRange, ADIndividuals, i, j);
        plvADFreqMAtrix(i,j) = mean(temp3);
        plvADRareMatrix(i,j) = mean(temp4);

        [temp5, temp6] = plv2Channels(MCI, fs, freqRange, MCIindividuals, i, j);
        plvMCIFreqMatrix(i,j) = mean(temp5);
        plvMCIRareMatrix(i,j) = mean(temp6);

        [~, pValFreq(i,j)] = ttest2(temp1, temp3);
        [~, pValRare(i,j)] = ttest2(temp2, temp4);
    end
end


%% P-Values

[h1, pValPLV_Freq] = ttest2(plvNormalFreq, plvADFreq);
[h2, pValPLV_Rare] = ttest2(plvNormalRare, plvADRare);

%% Box Plot - Histogram - Polar Histogram - Heatmap

% Data
disp('Normal-Rare: ')
for i = 1:normalIndividuals
    disp([num2str(i) '. ' num2str(plvNormalRare(i))]);
end

disp('Normal-Frequent: ')
for i = 1:normalIndividuals
    disp([num2str(i) '. ' num2str(plvNormalFreq(i))]);
end

disp('AD-Rare: ')
for i = 1:ADIndividuals
    disp([num2str(i) '. ' num2str(plvADRare(i))]);
end

disp('AD-Frequent: ')
for i = 1:ADIndividuals
    disp([num2str(i) '. ' num2str(plvADFreq(i))]);
end

% Box Plot
figure()

subplot(2,2,1)
boxplot(plvNormalFreq);
ylabel('Average PLV', 'Interpreter','latex')
title('Normal Individuals, Lemon Odor', 'Interpreter','latex')

subplot(2,2,2)
boxplot(plvNormalRare);
ylabel('Average PLV', 'Interpreter','latex')
title('Normal Individuals, Rose Odor', 'Interpreter','latex')

subplot(2,2,3)
boxplot(plvADFreq);
ylabel('Average PLV', 'Interpreter','latex')
title('AD Individuals, Lemon Odor', 'Interpreter','latex')

subplot(2,2,4)
boxplot(plvADRare);
ylabel('Average PLV', 'Interpreter','latex')
title('AD Individuals, Rose Odor', 'Interpreter','latex')

% Histogram
figure()

subplot(2,2,1)
histfit(plvNormalFreq, 20, "normal");
ylabel('Number of Participants', 'Interpreter','latex')
title('Normal Individuals, Lemon Odor', 'Interpreter','latex')

subplot(2,2,2)
histfit(plvNormalRare, 20, "normal");
ylabel('Number of Participants', 'Interpreter','latex')
title('Normal Individuals, Rose Odor', 'Interpreter','latex')

subplot(2,2,3)
histfit(plvADFreq, 20, "normal");
ylabel('Number of Participants', 'Interpreter','latex');
title('AD Individuals, Lemon Odor', 'Interpreter','latex')


subplot(2,2,4)
histfit(plvADRare, 20, "normal");
ylabel('Number of Participants', 'Interpreter','latex');
title('AD Individuals, Rose Odor', 'Interpreter','latex')

% PolarHistogram - Random Subject
figure()

subplot(2,2,1)
polarhistogram(phaseDifferenceNormalFreq, 20)
title('Normal Individuals, Lemon Odor Sub9', 'Interpreter','latex')

subplot(2,2,2)
polarhistogram(phaseDiffereneceNormalRare, 20)
title('Normal Individuals, Rose Odor Sub9', 'Interpreter','latex')

subplot(2,2,3)
polarhistogram(phaseDifferenceADFreq, 20)
title('AD Individuals, Lemon Odor Sub7', 'Interpreter','latex')

subplot(2,2,4)
polarhistogram(phaseDiffereneceADRare, 20)
title('AD Individuals, Rose Odor Sub7', 'Interpreter','latex')

% PolarHistogram - Mean
figure()

subplot(2,2,1)
polarhistogram(meanPhaseDifferenceNormalFreq, 20)
title('Normal Individuals, Lemon Odor', 'Interpreter','latex')

subplot(2,2,2)
polarhistogram(meanPhaseDiffereneceNormalRare, 20)
title('Normal Individuals, Rose Odor', 'Interpreter','latex')

subplot(2,2,3)
polarhistogram(meanPhaseDifferenceADFreq, 20)
title('AD Individuals, Lemon Odor', 'Interpreter','latex')

subplot(2,2,4)
polarhistogram(meanPhaseDiffereneceADRare, 20)
title('AD Individuals, Rose Odor', 'Interpreter','latex')

% Heatmap
figure()

subplot(2,2,1)
heatmap(plvNormalFreqMatrix);
colormap("winter");
caxis([0.5, 1]);
colorbar;
title('Normal Individuals, Lemon Odor')

subplot(2,2,2)
heatmap(plvNormalRareMatrix);
colormap("winter");
caxis([0.5, 1]);
colorbar;
title('Normal Individuals, Rose Odor')

subplot(2,2,3)
heatmap(plvADFreqMAtrix);
colormap("winter");
caxis([0.5, 1]);
colorbar;
title('AD Individuals, Lemon Odor')

subplot(2,2,4)
heatmap(plvADRareMatrix);
colormap("winter");
caxis([0.5, 1]);
colorbar;
title('AD Individuals, Rose Odor')

% P-Values
disp("P-Value of Frequent Smell in all channels:");
disp(pValFreq);
disp("P-Value of Rare Smell in all channels:");
disp(pValRare);

%% PLV for the 'MCI' datase and Figures wuith Differences with Normal and AD

% Calculate PLV for the 'MCI' dataset (Most Signifacnt Different CoupledChannels (1,2) (1,3) (1,4))

% channels(1,2)
[plvNormalFreq1, plvNormalRare1] = plv2Channels(normal, fs, freqRange, normalIndividuals, 1, 2);
[plvADFreq1, plvADRare1] = plv2Channels(AD, fs, freqRange, ADIndividuals, 1, 2);
[plvMCIFreq1, plvMCIRare1] = plv2Channels(MCI, fs, freqRange, MCIindividuals, 1, 2);

% channels(1,3)
[plvNormalFreq2, plvNormalRare2] = plv2Channels(normal, fs, freqRange, normalIndividuals, 1, 3);
[plvADFreq2, plvADRare2] = plv2Channels(AD, fs, freqRange, ADIndividuals, 1, 3);
[plvMCIFreq2, plvMCIRare2] = plv2Channels(MCI, fs, freqRange, MCIindividuals, 1, 3);

% channels(1,4)
[plvNormalFreq3, plvNormalRare3] = plv2Channels(normal, fs, freqRange, normalIndividuals, 1, 4);
[plvADFreq3, plvADRare3] = plv2Channels(AD, fs, freqRange, ADIndividuals, 1, 4);
[plvMCIFreq3, plvMCIRare3] = plv2Channels(MCI, fs, freqRange, MCIindividuals, 1, 4);

% Plotting
plottingPlv(plvNormalFreq1, plvNormalRare1, plvADFreq1, plvADRare1, plvMCIFreq1, plvMCIRare1)
plottingPlv(plvNormalFreq2, plvNormalRare2, plvADFreq2, plvADRare2, plvMCIFreq2, plvMCIRare2)
plottingPlv(plvNormalFreq3, plvNormalRare3, plvADFreq3, plvADRare3, plvMCIFreq3, plvMCIRare3)

%% P-Value of MCI

[pFreq12, tbFreq12] = anova_data(plvNormalFreq1, plvADFreq1, plvMCIFreq1);
[pRare12, tbRare12] = anova_data(plvNormalRare1, plvADRare1, plvMCIRare1);

[pFreq13, tbFreq13] = anova_data(plvNormalFreq2, plvADFreq2, plvMCIFreq2);
[pRare13, tbRare13] = anova_data(plvNormalRare2, plvADRare2, plvMCIRare2);

[pFreq14, tbFreq14] = anova_data(plvNormalFreq3, plvADFreq3, plvMCIFreq3);
[pRare14, tbRare14] = anova_data(plvNormalRare3, plvADRare3, plvMCIRare3);

%% Phase Difference of MCI - Random Subject

% channels(1,2)
[phaseDifferenceNormalFreq1, phaseDiffereneceNormalRare1] = avgPhase2Channels(normal, 1, 2, 9); % for 9th person
[phaseDifferenceADFreq1, phaseDiffereneceADRare1] = avgPhase2Channels(AD, 1,2, 7); % for 7th person
[phaseDifferenceMCIFreq1, phaseDiffereneceMCIRare1] = avgPhase2Channels(MCI, 1,2, 4); % for 4th person

% channels(1,3)
[phaseDifferenceNormalFreq2, phaseDiffereneceNormalRare2] = avgPhase2Channels(normal, 1, 2, 9); % for 9th person
[phaseDifferenceADFreq2, phaseDiffereneceADRare2] = avgPhase2Channels(AD, 1,2, 7); % for 7th person
[phaseDifferenceMCIFreq2, phaseDiffereneceMCIRare2] = avgPhase2Channels(MCI, 1,2, 4); % for 4th person

% channels(1,4)
[phaseDifferenceNormalFreq3, phaseDiffereneceNormalRare3] = avgPhase2Channels(normal, 1, 2, 9); % for 9th person
[phaseDifferenceADFreq3, phaseDiffereneceADRare3] = avgPhase2Channels(AD, 1,2, 7); % for 7th person
[phaseDifferenceMCIFreq3, phaseDiffereneceMCIRare3] = avgPhase2Channels(MCI, 1,2, 4); % for 4th person

% Plotting
plottingPhaseDiff(phaseDifferenceNormalFreq1, phaseDiffereneceNormalRare1, phaseDifferenceADFreq1, phaseDiffereneceADRare1, phaseDifferenceMCIFreq1, phaseDiffereneceMCIRare1)
plottingPhaseDiff(phaseDifferenceNormalFreq2, phaseDiffereneceNormalRare2, phaseDifferenceADFreq2, phaseDiffereneceADRare2, phaseDifferenceMCIFreq2, phaseDiffereneceMCIRare2)
plottingPhaseDiff(phaseDifferenceNormalFreq3, phaseDiffereneceNormalRare3, phaseDifferenceADFreq3, phaseDiffereneceADRare3, phaseDifferenceMCIFreq3, phaseDiffereneceMCIRare3)

%% Phase Difference of MCI - Mean

% channels(1,2)
[meanPhaseDifferenceNormalFreq1, meanPhaseDiffereneceNormalRare1] = meanPahseDiff(normal, 1, 2, normalIndividuals);
[meanPhaseDifferenceADFreq1, meanPhaseDiffereneceADRare1] = meanPahseDiff(AD, 1, 2, ADIndividuals);
[meanPhaseDifferenceMCIFreq1, meanPhaseDiffereneceMCIRare1] = meanPahseDiff(MCI, 1, 2, MCIindividuals);

% channels(1,3)
[meanPhaseDifferenceNormalFreq2, meanPhaseDiffereneceNormalRare2] = meanPahseDiff(normal, 1, 3, normalIndividuals);
[meanPhaseDifferenceADFreq2, meanPhaseDiffereneceADRare2] = meanPahseDiff(AD, 1, 3, ADIndividuals);
[meanPhaseDifferenceMCIFreq2, meanPhaseDiffereneceMCIRare2] = meanPahseDiff(MCI, 1, 3, MCIindividuals);

% channels(1,4)
[meanPhaseDifferenceNormalFreq3, meanPhaseDiffereneceNormalRare3] = meanPahseDiff(normal, 1, 4, normalIndividuals);
[meanPhaseDifferenceADFreq3, meanPhaseDiffereneceADRare3] = meanPahseDiff(AD, 1, 4, ADIndividuals);
[meanPhaseDifferenceMCIFreq3, meanPhaseDiffereneceMCIRare3] = meanPahseDiff(MCI, 1, 4, MCIindividuals);

% Plotting
plottingPhaseDiff(meanPhaseDifferenceNormalFreq1, meanPhaseDiffereneceNormalRare1, meanPhaseDifferenceADFreq1, meanPhaseDiffereneceADRare1, meanPhaseDifferenceMCIFreq1, meanPhaseDiffereneceMCIRare1)
plottingPhaseDiff(meanPhaseDifferenceNormalFreq2, meanPhaseDiffereneceNormalRare2, meanPhaseDifferenceADFreq2, meanPhaseDiffereneceADRare2, meanPhaseDifferenceMCIFreq2, meanPhaseDiffereneceMCIRare2)
plottingPhaseDiff(meanPhaseDifferenceNormalFreq3, meanPhaseDiffereneceNormalRare3, meanPhaseDifferenceADFreq3, meanPhaseDiffereneceADRare3, meanPhaseDifferenceMCIFreq3, meanPhaseDiffereneceMCIRare3)

%% Heatmap of MCI

figure()

subplot(1,2,1)
heatmap(plvMCIFreqMatrix);
colormap("winter");
caxis([0.5, 1]);
colorbar;
title('MCI Individuals, Lemon Odor')

subplot(1,2,2)
heatmap(plvMCIRareMatrix);
colormap("winter");
caxis([0.5, 1]);
colorbar;
title('MCI Individuals, Rose Odor')

%% MI data and Plot Diagrams

[meanMINormalFreq, meanMINormalRare] = meanMI(normal, channel1,channel2, normalIndividuals, freqRange, fs);
[meanMI_ADFreq, meanMI_ADRare] = meanMI(AD, channel1,channel2, ADIndividuals, freqRange, fs);

% PolarHistogram - Mean
figure()

subplot(2,2,1)
polarhistogram(meanMINormalFreq, 18)
title('Normal Individuals, Lemon Odor', 'Interpreter','latex')

subplot(2,2,2)
polarhistogram(meanMINormalRare, 18)
title('Normal Individuals, Rose Odor', 'Interpreter','latex')

subplot(2,2,3)
polarhistogram(meanMI_ADFreq, 18)
title('AD Individuals, Lemon Odor', 'Interpreter','latex')

subplot(2,2,4)
polarhistogram(meanMI_ADRare, 18)
title('AD Individuals, Rose Odor', 'Interpreter','latex')

max_length = max([length(meanMINormalFreq), length(meanMI_ADFreq)]);
meanMINormalFreq(end+1:max_length) = NaN;
meanMI_ADFreq(end+1:max_length) = NaN;

max_length = max([length(meanMINormalRare), length(meanMI_ADRare)]);
meanMINormalRare(end+1:max_length) = NaN;
meanMI_ADRare(end+1:max_length) = NaN;

freq = [meanMINormalFreq(:); meanMI_ADFreq(:)];
rare = [meanMINormalRare(:); meanMI_ADRare(:)];

groupLabels = repmat({'Normal', 'AD'}, [1, length(meanMINormalFreq)]);

figure()
% Box Plot
subplot(2, 2, 1);
boxplot(freq, groupLabels);
ylabel('Modulation Index (MI)');
title('Distribution of MI Values(Freq)');

% Scatter Plot
subplot(2, 2, 2);
scatter(1:length(meanMINormalFreq), meanMINormalFreq, 'b', 'filled');
hold on;
scatter(1:length(meanMI_ADFreq), meanMI_ADFreq, 'r', 'filled');
hold off;
xlabel('Subject');
ylabel('Modulation Index (MI)');
legend('Normal', 'AD');
title('Individual MI Values(Freq)');


subplot(2, 2, 3);
boxplot(rare, groupLabels);
ylabel('Modulation Index (MI)');
title('Distribution of MI Values(Rare)');

% Scatter Plot
subplot(2, 2, 4);
scatter(1:length(meanMINormalRare), meanMINormalRare, 'b', 'filled');
hold on;
scatter(1:length(meanMI_ADRare), meanMI_ADRare, 'r', 'filled');
hold off;
xlabel('Subject');
ylabel('Modulation Index (MI)');
legend('Normal', 'AD');
title('Individual MI Values(Rare)');

%% Functions

%function to calculate PLV
function plv = calculatePLVInRange(signal1, signal2, fs, freqRange)
    % Perform Fourier transform on the signals
    fft1 = fft(signal1);
    fft2 = fft(signal2);

    % Frequency axis
    f = (0:length(signal1)-1)*(fs/length(signal1));

    % Indices of frequencies within the specified range
    freqIndices = find(f >= freqRange(1) & f <= freqRange(2));

    % Extract the phase components of the signals within the frequency range
    phase1 = angle(fft1(freqIndices));
    phase2 = angle(fft2(freqIndices));

    % Calculate the Phase Locking Value (PLV)
    plv = abs(mean(exp(1i * (phase1 - phase2))));
end

% Function to calculate phase-locking values (PLV) for two channels in a dataset
function [plvFreq, plvRare] = plv2Channels(data, fs, freqRange, numIndividuals, channel1, channel2)
    % Inputs:
    %   - data: Structure array containing the data for different individuals.
    %   - fs: Sampling frequency of the data.
    %   - freqRange: Frequency range of interest for calculating the PLV.
    %   - numIndividuals: Number of individuals in the dataset.
    %   - channel1, channel2: Indices of the channels to calculate the PLV for.
    % Outputs:
    %   - plvFreq: Array of PLV values for the frequent trials of each individual.
    %   - plvRare: Array of PLV values for the rare trials of each individual.

    % Initialize output variables
    plvRare = zeros(1, numIndividuals); 
    plvFreq = zeros(1, numIndividuals);
    
    % Iterate over each individual in the dataset
    for i = 1:numIndividuals
        % Extract the epochs for the rare and frequent conditions for the current individual
        rareEpoch = data(i).epoch(:, :, data(i).odor == 1);
        frequentEpoch = data(i).epoch(:, :, data(i).odor == 0);
    
        % Extract the data for the selected channels from the rare and frequent epochs
        rareFz = squeeze(rareEpoch(channel1,:,:));
        rareCz = squeeze(rareEpoch(channel2,:,:));
    
        freqFz = squeeze(frequentEpoch(channel1,:,:));
        freqCz = squeeze(frequentEpoch(channel2,:,:));
    
        % Calculate the number of rare and frequent trials
        numRareTrials = size(rareFz,2);
        numFreqTrials = size(freqFz,2);
        
        % Iterate over the rare trials
        for j = 1:numRareTrials
            % Calculate the PLV between the two channels for the current rare trial
            plvRare(i) = plvRare(i) + calculatePLVInRange(rareFz(:,j), rareCz(:,j), fs, freqRange);
        end
        % Average the PLV values for the rare trials
        plvRare(i) = plvRare(i)/numRareTrials;
    
        % Iterate over the frequent trials
        for j = 1:numFreqTrials
            % Calculate the PLV between the two channels for the current frequent trial
            plvFreq(i) = plvFreq(i) + calculatePLVInRange(freqFz(:,j), freqCz(:,j), fs, freqRange);
        end
        % Average the PLV values for the frequent trials
        plvFreq(i) = plvFreq(i)/numFreqTrials;
    end
end

% Function to Calcukate Phase Difference for 2 Channels of Random Subject
function [averagePhaseFreq, averagePhaseRare] = avgPhase2Channels(data, channel1, channel2, subjectIndex)
    % Initialize average phase variables
    averagePhaseRare = 0;
    averagePhaseFreq = 0;

    % Extract rare and frequent epochs for the specified subject index
    rareEpoch = data(subjectIndex).epoch(:, :, data(subjectIndex).odor == 1);
    frequentEpoch = data(subjectIndex).epoch(:, :, data(subjectIndex).odor == 0);

    % Calculate average phase difference for rare trials
    numRareTrials = size(rareEpoch, 3);
    for i = 1:numRareTrials
        % Retrieve epoch data for the specified channels
        trialRare = rareEpoch(:,:,i);

        % Calculate phase difference between the two channels
        phaseData1 = angle(trialRare(channel1, :));
        phaseData2 = angle(trialRare(channel2, :));

        phase = rad2deg(phaseData1 - phaseData2);

        % Accumulate phase differences
        averagePhaseRare = averagePhaseRare + phase;
    end

    % Calculate average phase difference for rare trials
    averagePhaseRare = averagePhaseRare / numRareTrials;

    % Calculate average phase difference for frequent trials
    numFreqTrials = size(frequentEpoch,3);
    for i = 1:numFreqTrials
        % Retrieve epoch data for the specified channels
        trialFreq = frequentEpoch(:,:,i);

        % Calculate phase difference between the two channels
        phaseData1 = angle(trialFreq(channel1, :));
        phaseData2 = angle(trialFreq(channel2, :));

        phase = rad2deg(phaseData1 - phaseData2);

        % Accumulate phase differences
        averagePhaseFreq = averagePhaseFreq + phase;
    end

    % Calculate average phase difference for frequent trials
    averagePhaseFreq = averagePhaseFreq / numFreqTrials;
end

% Function to Calculate Mean Phase Difference
function [meanPhaseFreq, meanPhaseRare] = meanPahseDiff(data, channel1, channel2, numIndividuals)
    meanPhaseFreq = 0;  % Initialize mean phase difference for channel1 and channel2
    meanPhaseRare = 0;  % Initialize mean phase difference for channel2 and channel1

    % Iterate over the specified number of individuals
    for i = 1:numIndividuals
        [temp1, temp2] = avgPhase2Channels(data, channel1, channel2, i);  % Calculate average phase difference
        meanPhaseFreq = meanPhaseFreq + temp1;  % Accumulate phase differences for channel1 and channel2
        meanPhaseRare = meanPhaseRare + temp2;  % Accumulate phase differences for channel2 and channel1
    end
    
    meanPhaseFreq = meanPhaseFreq / numIndividuals;  % Compute mean phase difference for channel1 and channel2
    meanPhaseRare = meanPhaseRare / numIndividuals;  % Compute mean phase difference for channel2 and channel1
end

% Function to plot PLV
function plottingPlv(plvNormalFreq, plvNormalRare, plvADFreq, plvADRare, plvMCIFreq, plvMCIRare)

    % Box Plot
    figure()
    
    subplot(2,3,1)
    boxplot(plvNormalFreq);
    ylabel('Average PLV', 'Interpreter','latex')
    title('Normal Individuals, Lemon Odor', 'Interpreter','latex')
    
    subplot(2,3,4)
    boxplot(plvNormalRare);
    ylabel('Average PLV', 'Interpreter','latex')
    title('Normal Individuals, Rose Odor', 'Interpreter','latex')
    
    subplot(2,3,2)
    boxplot(plvADFreq);
    ylabel('Average PLV', 'Interpreter','latex')
    title('AD Individuals, Lemon Odor', 'Interpreter','latex')
    
    subplot(2,3,5)
    boxplot(plvADRare);
    ylabel('Average PLV', 'Interpreter','latex')
    title('AD Individuals, Rose Odor', 'Interpreter','latex')

    subplot(2,3,3)
    boxplot(plvMCIFreq);
    ylabel('Average PLV', 'Interpreter','latex')
    title('MCI Individuals, Lemon Odor', 'Interpreter','latex')

    subplot(2,3,6)
    boxplot(plvMCIRare);
    ylabel('Average PLV', 'Interpreter','latex')
    title('MCI Individuals, Rose Odor', 'Interpreter','latex')

    % Histfit
    figure()
    
    subplot(2,3,1)
    histfit(plvNormalFreq, 20, "normal");
    ylabel('Number of Participants', 'Interpreter','latex')
    title('Normal Individuals, Lemon Odor', 'Interpreter','latex')
    
    subplot(2,3,4)
    histfit(plvNormalRare, 20, "normal");
    ylabel('Number of Participants', 'Interpreter','latex')
    title('Normal Individuals, Rose Odor', 'Interpreter','latex')
    
    subplot(2,3,2)
    histfit(plvADFreq, 20, "normal");
    ylabel('Number of Participants', 'Interpreter','latex')
    title('AD Individuals, Lemon Odor', 'Interpreter','latex')
    
    subplot(2,3,5)
    histfit(plvADRare, 20, "normal");
    ylabel('Number of Participants', 'Interpreter','latex')
    title('AD Individuals, Rose Odor', 'Interpreter','latex')

    subplot(2,3,3)
    histfit(plvMCIFreq, 20, "normal");
    ylabel('Number of Participants', 'Interpreter','latex')
    title('MCI Individuals, Lemon Odor', 'Interpreter','latex')

    subplot(2,3,6)
    histfit(plvMCIRare, 20, "normal");
    ylabel('Number of Participants', 'Interpreter','latex')
    title('MCI Individuals, Rose Odor', 'Interpreter','latex')
end

% Function to calculate P-Value of 3 groups of data
function [p, tb] = anova_data(data1, data2, data3)
    % Pad the shorter data sets with NaN values
    max_length = max([length(data1), length(data2), length(data3)]);
    data1(end+1:max_length) = NaN;
    data2(end+1:max_length) = NaN;
    data3(end+1:max_length) = NaN;
    
    % Use anova1 function to compare the three data sets
    [p, tb] = anova1([data1', data2', data3']);
end

% Function to plot Phase Difference for Random Subject
function plottingPhaseDiff(phaseDifferenceNormalFreq, phaseDiffereneceNormalRare, phaseDifferenceADFreq, phaseDiffereneceADRare, phaseDifferenceMCIFreq, phaseDiffereneceMCIRare)
    
    figure()
    
    subplot(2,3,1)
    polarhistogram(phaseDifferenceNormalFreq, 20)
    title('AD Individuals, Rose Odor Sub9', 'Interpreter','latex')
    
    subplot(2,3,4)
    polarhistogram(phaseDiffereneceNormalRare, 20)
    title('AD Individuals, Rose Odor Sub9', 'Interpreter','latex')
    
    subplot(2,3,2)
    polarhistogram(phaseDifferenceADFreq, 20)
    title('AD Individuals, Rose Odor Sub7', 'Interpreter','latex')
    
    subplot(2,3,5)
    polarhistogram(phaseDiffereneceADRare, 20)
    title('AD Individuals, Rose Odor Sub7', 'Interpreter','latex')

    subplot(2,3,3)
    polarhistogram(phaseDifferenceMCIFreq, 20)
    title('AD Individuals, Rose Odor Sub4', 'Interpreter','latex')

    subplot(2,3,6)
    polarhistogram(phaseDiffereneceMCIRare, 20)
    title('AD Individuals, Rose Odor Sub4', 'Interpreter','latex')
end

% Function to Calculate MI
function modulation_index = calculate_MI_PAC(signal_low_freq, signal_high_freq, num_bins, freq_range, fs)
    % Filter the low-frequency signal
    low_freq_filtered = bandpass(signal_low_freq, freq_range, fs);

    % Compute the Hilbert transform of the low-frequency signal
    low_freq_hilbert = hilbert(low_freq_filtered);

    % Calculate the phase of the low-frequency signal
    low_freq_phase = angle(low_freq_hilbert);

    % Filter the high-frequency signal
    high_freq_filtered = bandpass(signal_high_freq, freq_range, fs);

    % Calculate the amplitude of the high-frequency signal
    high_freq_amplitude = abs(hilbert(high_freq_filtered));

    % Calculate the modulation index
    phase_bins = linspace(-pi, pi, num_bins+1);
    bin_indices = discretize(low_freq_phase, phase_bins);
    modulation_index = zeros(1, num_bins);

    for bin_idx = 1:num_bins
        indices = bin_indices == bin_idx;
        modulation_index(bin_idx) = mean(high_freq_amplitude(indices));
    end

    modulation_index = modulation_index';

end

% Function to Calculate MI between to channels
function [MIFreq, MIRare] = MI2Channels(data, channel1, channel2, subjectIndex, freqRange, fs)
    % Initialize average phase variables
    MIFreq = 0;
    MIRare = 0;

    % Extract rare and frequent epochs for the specified subject index
    rareEpoch = data(subjectIndex).epoch(:, :, data(subjectIndex).odor == 1);
    frequentEpoch = data(subjectIndex).epoch(:, :, data(subjectIndex).odor == 0);

    % Calculate average phase difference for rare trials
    numRareTrials = size(rareEpoch, 3);
    for i = 1:numRareTrials
        % Retrieve epoch data for the specified channels
        trialRare = rareEpoch(:,:,i);
        
        MI = calculate_MI_PAC(trialRare(channel1,:), trialRare(channel2,:), 18, freqRange, fs);

        % Accumulate phase differences
        MIRare = MIRare + MI;
    end

    % Calculate average phase difference for rare trials
    MIRare = MIRare / numRareTrials;

    % Calculate average phase difference for frequent trials
    numFreqTrials = size(frequentEpoch,3);
    for i = 1:numFreqTrials
        % Retrieve epoch data for the specified channels
        trialFreq = frequentEpoch(:,:,i);
        
        MI = calculate_MI_PAC(trialFreq(channel1,:), trialFreq(channel2,:), 18, freqRange, fs);

        % Accumulate phase differences
        MIFreq = MIFreq + MI;
    end

    % Calculate average phase difference for frequent trials
    MIFreq = MIFreq / numFreqTrials;
end

% Function to Calculate Mean MI
function [meanMIFreq, meanMIRare] = meanMI(data, channel1, channel2, numIndividuals, freqRange, fs)
    meanMIFreq = 0;  % Initialize mean phase difference for channel1 and channel2
    meanMIRare = 0;  % Initialize mean phase difference for channel2 and channel1

    % Iterate over the specified number of individuals
    for i = 1:numIndividuals
        [temp1, temp2] = MI2Channels(data, channel1, channel2, i, freqRange, fs);  % Calculate average phase difference
        meanMIFreq = meanMIFreq + temp1;  % Accumulate phase differences for channel1 and channel2
        meanMIRare = meanMIRare + temp2;  % Accumulate phase differences for channel2 and channel1
    end
    
    meanMIFreq = meanMIFreq / numIndividuals;  % Compute mean phase difference for channel1 and channel2
    meanMIRare = meanMIRare / numIndividuals;  % Compute mean phase difference for channel2 and channel1
end
