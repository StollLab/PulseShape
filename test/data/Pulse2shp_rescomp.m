%% calculate pulse shapes with easyspin and export to Bruker SpinJet format 
% Eric G. B. Evans - October 2020
% Include resonantor compensation 
% Include time resoltuion of the AWG
clear
close

%% Observer pulse definition

Par{1}.Type = 'gaussian';  % pulse shape
Par{1}.trunc = 0.1; % guassian pulse truncation (0 to 1)
Par{1}.tp = 0.06;  % pulse length, 탎
Par{1}.Phase = 0;  % phase of the pulse in rad
Par{1}.Flip = pi;  % pulse flip angle
Par{1}.TimeStep = 0.000625;    % Time resolution of the AWG SpinJet in 탎

%% Pump pulse definition

Par{2}.Type = 'sech/tanh'; % pulse shape
% Par{2}.trise = 0.030; % for Quartersin: rise time, 탎
%Par{2}.n = 1; % increase for "higher order" sech/tanh pulses
% Par{2}.trunc = 0.1; % guassian pulse truncation (0 to 1)
% Par{2}.nwurst = 1; steepness parameter for WURST
Par{2}.beta = 10;  % truncation parameter for sech/tanh or WURST
Par{2}.tp = 0.150; % pulse length, 탎
Par{2}.Phase = 0; % phase of the pulse in rad
Par{2}.Flip = pi; % pulse flip angle
Par{2}.Frequency = [40 120]; % Frequency bandwidth (include offset for DEER pump pulses (i.e. [40 120])
Par{2}.TimeStep=0.000625; % Time resolution of the AWG SpinJet in 탎

%% Resonator compensation 

filename = 'Transferfunction.dat';
delimiter = ' ';
formatSpec = '%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string');
fclose(fileID);

xresponse = dataArray{:, 1};
yresponse = dataArray{:, 2};

Par{1}.FrequencyResponse = [xresponse,yresponse];
Par{2}.FrequencyResponse = [xresponse,yresponse];


prompt = {'Enter frequency (LO) in GHz:','Enter shape number between 10 and 1e6:'};
dlgtitle = 'Input Frequency and Shape Number';
definput = {'33.800','10'};
dims = [1 60];
userinput = inputdlg(prompt,dlgtitle,dims,definput);    %prompt user for LO frequency and shape #
Par{1}.mwFreq = str2double(userinput{1});   % frequency of observer pulse (LO), in GHz
Par{2}.mwFreq = str2double(userinput{1});   % frequency of observer pulse (LO), in GHz
shapeNumbr = userinput{2};

clearvars filename delimiter formatSpec fileID dataArray ans;
%pulse(Par{2});
%% Pulse calculation and plotting

[t{1},IQ{1},mod{1}] = pulse(Par{1}); % calculate observer pulse
[t{2},IQ{2},mod{2}] = pulse(Par{2}); % calculate pump pulse
range{1} = linspace(-40,80,256);
[offsets{1},M{1}] = exciteprofile(t{1},IQ{1},range{1}); % observer excitation profile
[offsets{2},M{2}] = exciteprofile(t{2},IQ{2}); % observer excitation profile

plt = tiledlayout(1,3);
set(gcf,'position',[20,100,1200,300]);

ax1 = nexttile;
plot(t{1},IQ{1},t{2},real(IQ{2}),'LineWidth',2);
title(ax1, 'Amplitude modulation');
xlabel(ax1, 'time (us)');
ylabel(ax1, 'nutation frequency (MHz)');
legend(ax1, 'observer','pump');

ax2 = nexttile;
plot(t{1},mod{1}.freq,t{2},mod{2}.freq,'LineWidth',2);
title(ax2, 'Frequency modulation');
xlabel(ax2, 'time (us)');
ylabel(ax2, 'frequency offset (MHz)');
legend(ax2, 'observer','pump');

ax3 = nexttile;
plot(offsets{1},M{1}(3,:),offsets{2},M{2}(3,:),'LineWidth',2);
title(ax3, 'Excitation profile');
xlabel(ax3, 'frequency offset (MHz)');
ylabel(ax3, 'MZ / M0');
legend(ax3, 'observer','pump');

%% Pump pulse shape normalization for export
%      
    result(:,1)=real(IQ{2}); % seperate real part
    result(:,2)=imag(IQ{2}); % seperate imag part 
    
    % normalization of pulse shape
    
    pulsemax_real = max(abs(result(:,1)));
    pulsemax_imag = max(abs(result(:,2)));
    
    if pulsemax_real >= pulsemax_imag
        pulse_norm = result;  % ./pulsemax_real;
    else
        pulse_norm = result;  % ./pulsemax_imag;
    end
    
%% Create and save .shp file and save pulse figure as .eps

str1 = 'begin shape';
str2 = 'end shape';
quote = ' "';
endquote = '"\n';

[filename,pathname] = uiputfile('*.shp','Save file as');
path_file = fullfile(pathname,filename);
fileID1 = fopen(path_file,'w');

plotname = strrep(filename,'.shp','.eps');
exportgraphics(plt,plotname); %save plot as .eps

header = strcat(str1,shapeNumbr,quote,filename,endquote);   %prepare header text
footer = strcat(str2,shapeNumbr);   %prepare footer text

format ='%.5e,%.5e\n';  % style of shp file 
datapoints = length(t{2});
array = zeros(2*datapoints,1);

fprintf(fileID1,header);
    for m = 1:datapoints     
        array(m*2-1)=pulse_norm(m,1); 
        array(m*2)=pulse_norm(m,2);
    end
    fprintf(fileID1,format,array);  %write the data into the two columns of the shp file
fprintf(fileID1,footer);