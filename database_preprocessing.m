% MATLAB script to save data from each patient in a single .csv file to be 
% used later in the Python script

% also the classification of EEG signals according to 2 evaluations, SAM 
% and HAM-A, is followed, the results being added to the .csv file as 
% labels

% the script is made in general mode and can be applied in turn for each 
% subject in the database, only needing to change the name of the file to 
% be loaded, which is done using the 'find and replace' function provided 
% by matlab

% it is necessary to apply the script on all the files in the database in 
% order to obtain a complete .csv file containing basically only the EEG 
% signal and the corresponding label/diagnosis

close all
clc

% importing files containing medical data

load('S01.mat')

% definition of labels for the classification process

% label resulting from the SAM experiment

avg_valence = mean(labels(:, 1));
avg_arousal = mean(labels(:, 2));

if ((avg_valence < 5) && (avg_arousal > 5) == 0)
    SAM_label = "NORMAL";

elseif (avg_valence >= 0 && avg_valence <= 2) && (avg_arousal >= 7 && avg_arousal <= 9)
    SAM_label = "SEVERE_ANXIETY";

elseif (avg_valence >= 2 && avg_valence <= 4) && (avg_arousal >= 6 && avg_arousal <= 7)
    SAM_label = "MODERATE_ANXIETY";

elseif (avg_valence >= 4 && avg_valence <= 5) && (avg_arousal >= 5 && avg_arousal <= 6)
    SAM_label = "LIGHT_ANXIETY";

else
    SAM_label = "NORMAL";

end

% label resulting from the completion of the HAM-A questionnaire

avg_hamilton = 0.35 * hamilton(1, 1) + 0.65 * hamilton(1, 2);

if avg_hamilton > 0 && avg_hamilton <= 12
    HAM_A_label = "NORMAL";

elseif avg_hamilton > 12 && avg_hamilton <= 20
    HAM_A_label = "LIGHT_ANXIETY";

elseif avg_hamilton > 20 && avg_hamilton <= 25
    HAM_A_label = "MODERATE_ANXIETY";

elseif avg_hamilton > 25
    HAM_A_label = "SEVERE_ANXIETY";

end

% storing data as a table

S01_data_cell = cell(14, 23043);

for i = 1 : 14
    
    for j = 1 : 23040

        S01_data_cell{i, j} = data(i, j);
    
    end

end

electrodes = ["S01_AF3" "S01_AF4" "S01_F3" "S01_F4" "S01_FC5" "S01_FC6" "S01_F7" "S01_F8" "S01_T7" "S01_T8" "S01_P7" "S01_P8" "S01_O1" "S01_O2"];

for i = 1 : length(electrodes)

    S01_data_cell{i, 23041} = electrodes(i);

end

for j = 1 : length(electrodes)

    S01_data_cell{j, 23042} = SAM_label;

end

for k = 1 : 14

    S01_data_cell{k, 23043} = HAM_A_label;

end

S01_data = cell2table(S01_data_cell);

% example of EEG signal

figure();
sensor = randsample(1 : 14, 1);
x = table2array(S01_data(sensor, 1 : 23040));
plot(x);
title(S01_data_cell{sensor, 23041});
xlabel('Time');
ylabel('Amplitude');

% saving data as a .csv file

save_location = 'C:\DODDY\PROIECT LICENȚĂ - ANXIETY DETECTION\EEG_DATABASE\Raw EEG Database\Database .csv files';
table_path_format = fullfile(save_location, 'S01_data.csv');
writetable(S01_data, table_path_format, 'WriteVariableNames', true);

% in order to obtain the complete database it is necessary to merge all 
% the .csv files resulting from the MATLAB script










