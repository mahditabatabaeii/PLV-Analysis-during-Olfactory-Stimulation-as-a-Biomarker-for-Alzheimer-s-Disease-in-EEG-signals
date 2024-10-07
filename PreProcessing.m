%% Load Subject1

% Load the data from the specified path
data1 = load('Dataset\Preprocess\Subject1.mat');

% Convert the data table to an array and transpose it
subject1 = table2array(data1.subject1)';

% Remove the 20th channel from the data
subject1 = subject1([1:19],:);

%% Load Subject2

% Load the data from the specified path
data2 = load('Dataset\Preprocess\Subject2.mat');

% Convert the data table to an array and transpose it
subject2 = table2array(data2.subject2)';

% Remove the 20th channel from the data
subject2 = subject2([1:19],:);

%% Plot Frequency Spectrum

% Extract data from EEG structure
data = EEG.data;

% Apply FFT to the Fz channel data
[f, P1] = fftFunc(data(5,:), 200);

% Plot the frequency spectrum
plot(f, P1);

%% Epoch Data Subject1

% Extract epochs for Subject 1 using the specified parameters
data_to_epoch = EEG.data;

epochDuration = 3;          % Duration of each epoch in seconds
samplingRate = 200;         % Sampling rate in Hz
numTrials = 120;            % Number of trials

initTimeSub1 = 14;          % Initial time for Subject 1 in seconds

epoches1 = epochData(data_to_epoch, initTimeSub1, epochDuration, samplingRate, numTrials);

%% Epoch Data Subject2

% Extract epochs for Subject 2 using the specified parameters
data_to_epoch = EEG.data;

epochDuration = 3;          % Duration of each epoch in seconds
samplingRate = 200;         % Sampling rate in Hz
numTrials = 120;            % Number of trials

initTimeSub1 = 16.4;          % Initial time for Subject 1 in seconds

epoches2 = epochData(data_to_epoch, initTimeSub1, epochDuration, samplingRate, numTrials);

%% Remove Channels Subject1

epochedData = EEG.data;
filePath1 = 'Dataset\Preprocess\Subject1-Preprocessed.mat';
selectedEpochData1 = removeChannels(epochedData, filePath1);

%% Remove Channels Subject1

epochedData = EEG.data;
filePath2 = 'Dataset\Preprocess\Subject1-Preprocessed.mat';
selectedEpochData2 = removeChannels(epochedData, filePath2);

%% Struct subject1

% Define the indices of the noisy trials for Subject 1
noisyTrials1 = [40 45 53 94];

% Specify the file path to save the struct for Subject 1
filePath1 = 'Dataset\Preprocess\Subject1-Struct.mat';

% Create the struct for Subject 1 using the makeStruct function
mystruct1 = makeStruct(selectedEpochData1, noisyTrials1, filePath1);

%% Struct Subject2

% Define the indices of the noisy trials for Subject 2
noisyTrials2 = [1 2 3 6 8 9 26 27 30 72 73 74 75 76 79 81 82 85 86 88 92 96 100 101 102 104 109 110 111 112 113 114 115 116 117 120];

% Specify the file path to save the struct for Subject 2
filePath2 = 'Dataset\Preprocess\Subject2-Struct.mat';

% Create the struct for Subject 2 using the makeStruct function
mystruct2 = makeStruct(selectedEpochData2, noisyTrials2, filePath2);

%% functions 

% Function to calculate the one-sided frequency spectrum using FFT
function [f, P1] = fftFunc(X, Fs)
    L = length(X); % Length of the input data
    Y = fft(X); % Perform the FFT
    P2 = abs(Y/L); % Two-sided spectrum
    P1 = P2(1:L/2 + 1); % One-sided spectrum
    P1(2:end-1) = 2*P1(2:end-1); % Multiply by 2 (except the DC component and Nyquist frequency)
    f = (0:(L/2))*Fs/L; % Frequency axis
end

% Function to epoch the data based on specified parameters
function epoches = epochData(data, initTime, epochDuration, samplingRate, numTrials)
    
    epochSamples = epochDuration * samplingRate; % Number of samples per epoch (600)
    
    numChannels = size(data, 1); % Number of channels (19)
    
    epoches = zeros(numChannels, epochSamples, numTrials); % Initialize the epoch data matrix
    
    initSample = initTime * samplingRate; % Initial sample index
    
    % Iterate over channels, trials, and samples to extract epochs
    for i = 1:numChannels
        for j = 1:numTrials
            % Calculate the start and end indices of the epoch
            startSample = initSample + (samplingRate * 10 * j) - epochSamples + 1;
            endSample = initSample + (samplingRate * 10 * j);
            
            % Extract the data for the current epoch
            epoches(i, :, j) = data_to_epoch(i, startSample:endSample);
        end
    end
end

% Function to remove unwanted channels from the epoched data and save the result
function selectedEpochData = removeChannels(epochedData, filePath)
    % Input:
    % - epochedData: The epoched data in the format [NumChannels x NumSamples x NumTrials]
    % - filePath: The file path to save the processed data

    channelsIdx = ismember(1:19, [1, 5, 10, 15]);  % Indices of the desired channels
    
    selectedEpochData = epochedData(channelsIdx, :, :);  % Select the desired channels
    
    save(filePath, 'selectedEpochData');  % Save the processed data
end

% Function to create a struct and assign matrices as fields
function myStruct = makeStruct(selectedEpochData, noisyTrials, filePath)
    % Input:
    % - selectedEpochData: The selected epoch data matrix
    % - noisyTrials: The indices of the noisy trials
    % - filePath: The file path to save the struct

    % Create a struct and assign the matrices as fields
    myStruct = struct();
    
    % Adding the clean epoch data to the struct
    myStruct.cleanEpochData = selectedEpochData;
    
    % Adding the odor from normal data to the struct 
    odor = load("Dataset\Normal.mat").normal(2).odor;
    myStruct.odor = odor; % Num_trials x 1
    
    % Adding the noisy trials to the struct 
    myStruct.noisy = noisyTrials.'; % Num_noisy x 1
    
    % Save the struct to the specified file path
    save(filePath, 'myStruct');
end


