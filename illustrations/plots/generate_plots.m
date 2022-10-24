%% Settings

% in cm
plot_height = 8
plot_width = 12
include_origo = 0

save_format = 'pdf';
filename_model1_training_cost = 'training_cost';
filename_model1_validation_cost = 'validation_cost';
filename_model1_word2vec = 'word2vec';
filename_model2_perplexity= 'cond_perplexity';

files = {filename_model1_training_cost, filename_model1_validation_cost, ...
    filename_model1_word2vec, filename_model2_perplexity}

% At the end of this script there is code that crop the generated PDF:s
crop_margins = 10


%% Model 1 --- Cost on training set for different number of layers
clf, clc

cost_t1 = [5.4617494759341785, 4.4984070821849222, 4.0239650144625436, 3.6942282601758309, 3.4695632107342562, 3.316313056442338, 3.2072157259713574, 3.1255917073361159, 3.0627755642808632, 3.0128819494876766, 2.9747961249182069, 2.9428385019060319, 2.916064008528811, 2.8956999908292356, 2.8785876749881028, 2.8638150137499503, 2.8522819745213854, 2.8423323373552507, 2.8336097887687877, 2.8291611656537516, 2.8226051062569399, 2.818645756910295, 2.8179572571110603, 2.8165069121539896, 2.8157273407756978]; %, 2.816821289372323];
%cost_v1 = [4.9192881256943446, 4.8013820375215026, 4.8800325642157034, 4.9931207415379513, 5.11343015058325, 5.2286882285021861, 5.3387207423218896, 5.4371651017775227, 5.5326749245179903, 5.6164871915764767, 5.6933991318448971, 5.7653322104357798, 5.8226366515553325, 5.8819670958912704, 5.9346590206601206, 5.9859015732511471, 6.037446798622061, 6.0767728977028383, 6.1272329011969608, 6.1673016245430761, 6.1985645875143351, 6.2335917565581997, 6.2625389393097768, 6.2962588752956563, 6.3309865577067805] %, 6.364217030936425]

cost_t2 = [5.7813608015805933, 4.7766867418628056, 4.3381103831644587, 4.0188330851133705, 3.7730422702537574, 3.5790110812889138, 3.4257695121958776, 3.3009048339533926, 3.1982337750275125, 3.111994492429162, 3.0397105410813077, 2.978878535546627, 2.9259525896256346, 2.8804216087070214, 2.8412240713671379, 2.8072833341801831, 2.7765699039207496, 2.7485399801186499, 2.724376695816892, 2.7018453527924979, 2.6831717354246809, 2.6666487910856449, 2.6500570900234472, 2.6355989293132338, 2.6227548882344047]; %, 2.6125852732392132];
%cost_v2 = [5.1445517135760106, 4.9323542911634535, 4.9391294846840958, 5.0319051424078989, 5.1658684686783261, 5.3002700721670726, 5.4462876514119838, 5.5910153114248855, 5.7282969735521787, 5.848917925073466, 5.9607748497079269, 6.0712393720434346, 6.1768071053881162, 6.2678333135482367, 6.3587222374032395, 6.4472393854823684, 6.5227633638994407, 6.6030812338732794, 6.6647173344323392, 6.7273194703268349, 6.7986598471545294, 6.8529349342836152, 6.9166732368119268, 6.9694796444954132, 7.0128912353515629 ]% ,7.0579957734554188]

cost_t3 = [6.1246144142731795, 5.1344013523159902, 4.7581727708535748, 4.4804483627086968, 4.2666882144521336, 4.0889972131409618, 3.9381548784130116, 3.8075347953060557, 3.7114197829193269, 3.628406627829305, 3.5504357366416661, 3.481468450710858, 3.4212911505529724, 3.3700021919986316, 3.323691906924175, 3.287586515010311, 3.2567560937561963, 3.225489418145969, 3.1886928799237091, 3.1657345480362169, 3.1406597443401512, 3.1204150446393162, 3.1158013428044198, 3.0912756020792851, 3.0780452724398697]; %, 3.0663568629540769];
%cost_v3 = [5.4560073880536839, 5.1734333787270641, 5.1267717273957141, 5.2047306585749356, 5.2973209654081854, 5.4046294438073392, 5.5182203828304184, 5.6178329215793434, 5.7274946356257166, 5.8297189695025802, 5.9399317246183339, 6.0776334822068518, 6.1639477875035835, 6.2758562840452985, 6.4062457499372849, 6.3933932579110522, 6.442797073224269, 6.5313607480110383, 6.5844163863155822, 6.6662584329307624, 6.7116733040065943, 6.7686747587711436, 6.8296061216581849, 6.9114717144047448, 6.9888084873584431]%,7.0320798261449973]

x = linspace(1, length(cost_t1), length(cost_t1));
%set(hFig, 'Position', )
%figure
plot(x, cost_t1, '-o', x, cost_t2, '-x', x, cost_t3, '-^', 'LineWidth', 2)
grid on

title('Cost on training set')
xlabel('Epoch')
ylabel('Cost')
if include_origo
    xlim([0, x(end)])
    ylim([0, max([cost_t1(1) cost_t2(1) cost_t3(1)])])
end
legend('1 layer', '2 layers', '3 layers')

set(gcf,'units','centimeters','position',[0, 0, plot_width, plot_height])
pause(1)
saveas(gcf, filename_model1_training_cost, save_format)
%% Model 1 --- Cost on validation set for different number of layers with 0.5 dropout

%cost_t1 = [6.0016835779490201, 5.3096092194687898, 5.0799916279594308, 4.9480455563927661, 4.858224174848063, 4.7920583870979732, 4.7385028390642354, 4.6967251706195965, 4.6618628036576482, 4.6321187093512055, 4.606031452779237, 4.582975354344712, 4.5630176315694895, 4.5452524694452432, 4.5286192433313666, 4.5139338919547605, 4.5001157773400324, 4.4885522859060218, 4.4773771015302781, 4.4660838997932863, 4.4570460993577985, 4.4470371186696935, 4.4389424629017791, 4.4310510413465165, 4.4230305047204652];
cost_v1 = [5.2597913353596262, 4.9926503088714878, 4.881718724802, 4.8210594079253868, 4.7851068591196606, 4.7614982059023792, 4.7455698387775946, 4.7346566016520928, 4.7271328147398224, 4.7206306079549529, 4.7196530095371632, 4.7170925091384746, 4.7167001622751217, 4.713871031594933, 4.7143371386046802, 4.7181521886423097, 4.7178350830078122, 4.716832250192625, 4.7177254661070096, 4.7201099668730286, 4.7217549399279672, 4.7226071586958858, 4.7239097329235955, 4.7249071145713879, 4.7278521728515628];
%cost_t2 = [6.3022522332051079, 5.5453854989763443, 5.2886931310353544, 5.1390389212206538, 5.0385371705650677, 4.9632300825554712, 4.9037352663611395, 4.8554446229886281, 4.8157109495831021, 4.7812027947285456, 4.7514619044579831, 4.7246680550357416, 4.7011398165165472, 4.6796618624459665, 4.6617452428207784, 4.6439158594741432, 4.6268686376271511, 4.6125008532528948, 4.5993274811875393, 4.5856913204289933, 4.5736909179687499, 4.5622478657833812, 4.551610097352623, 4.5433274057456083, 4.5334017598883154];
cost_v2 = [5.5181325985969751, 5.1862212490816733, 5.0361228648894425, 4.9516990619624428, 4.9011458398661478, 4.86556620186622, 4.8397007737465954, 4.8216005349815436, 4.8100206035649009, 4.7976882626594755, 4.7911256478685855, 4.7832108327883098, 4.7798699755187428, 4.7765800518070884, 4.7740426943717749, 4.7716393119042069, 4.7703030003538922, 4.7680622485799526, 4.7683248145427175, 4.7683870648025373, 4.7670931586869267, 4.76782965423864, 4.7659167116497638, 4.7639266519808992, 4.7668371106069021];

%cost_t3 = [6.4816051220579194, 5.7328703749603429, 5.4646965025306349, 5.3093979180815261, 5.2015150526017706, 5.1228590754010348, 5.0609956954723687, 5.0111017302692238, 4.9694704603785791, 4.9349274854321168, 4.9046419885315871, 4.8783502256132021, 4.8551235018502634, 4.8348150120459232, 4.8157452662124243, 4.7993693570364551, 4.7835492501041008, 4.7699359987520324, 4.7558279028568169, 4.7448816836594325, 4.7340673623641738, 4.7229410700919061, 4.7131488978971685, 4.7048000178458125, 4.6950367135759539];
cost_v3 = [5.7279108561944527, 5.3653675660299598, 5.1969510419653098, 5.0978705869902168, 5.032449853179652, 4.9898257670271287, 4.9587054527352707, 4.9360085667601421, 4.91696121635787, 4.9044228762005444, 4.8931933733738893, 4.8819584109805048, 4.8772785263761467, 4.8700473610414274, 4.8654562881889696, 4.8618304191379371, 4.8599184032755165, 4.8562756711627362, 4.8555149379345259, 4.8524309490798814, 4.8505778601410192, 4.8515128487403238, 4.8494187801256095, 4.8472703650238316, 4.8488767459414417];

x = linspace(1, length(cost_v1), length(cost_v1));
figure
plot(x, cost_v1, '-o', x, cost_v2, '-x', x, cost_v3, '-^', 'LineWidth', 2)
grid on

title('Cost on validation set with 0.5 dropout')
xlabel('Epoch')
ylabel('Cost')
if include_origo
    xlim([0, x(end)])
    ylim([0, max([cost_v1(1) cost_v2(1) cost_v3(1)])])
end
legend('1 layer', '2 layers', '3 layers')
set(gcf,'units','centimeters','position',[0, 0, plot_width, plot_height])
pause(1)
saveas(gcf, filename_model1_validation_cost, save_format)
%% Model 1 ---Cost on training set

cost_t_with_w2v = [5.4656879724802705, 4.5038821288752677, 4.0313249826189228, 3.7016608361568548, 3.4770344960169139, 3.3241730297107988, 3.2144585410181037, 3.1318528783188255, 3.0685672508278472, 3.0199830402819638, 2.9798737052491475, 2.9493995217260371, 2.9231220803817517, 2.9011126888681789, 2.8845115830474697];
%cost_v_with_vw2v = [4.9223729292843323, 4.8041556934041711, 4.8768100787521504, 4.9915380215426106, 5.1108531930905965, 5.2246349285720681, 5.3370094537297517, 5.427589340909905, 5.5230753319416568, 5.6092448159314081, 5.6848202150677318, 5.754875365091026, 5.8250742389084005, 5.879251742581709, 5.938603168452552];
cost_t_without_w2v = [5.6341443462855922, 4.6288593934344764, 4.1516133628883942, 3.8147650846684646, 3.5776042086993378, 3.4113469746391178, 3.2915698556657973, 3.200998428577094, 3.1303742317935539, 3.0754120441572317, 3.0307356333466351, 2.9951529457363382, 2.964810083844335, 2.9395402649235605, 2.9188008631982174]; %, 2.9017037965416304]
%cost_v_without_w2v = [5.0303356653615969, 4.8669230441872129, 4.9157966543775089, 5.0153963953420657, 5.1347124740180616, 5.2442929637104001, 5.3478585171480795, 5.4460194998924889, 5.5359097654010174, 5.620532892909619, 5.7034140154637329, 5.7679498459002296, 5.8360943547520066, 5.8941633311980359, 5.9514241405802037] %, 5.9938010399494699]

x = linspace(1, length(cost_t_with_w2v), length(cost_t_with_w2v));
figure
plot(x, cost_t_with_w2v, '-o', x, cost_t_without_w2v, '-x', 'LineWidth', 2)
grid on

title('Cost on training set')
xlabel('Epoch')
ylabel('Cost')

if include_origo
    xlim([0, x(end)])
    ylim([0, max([cost_t_with_w2v(1) cost_t_without_w2v(1)])])
end

legend('initialized with word2vec', 'random uniform initializer')
set(gcf,'units','centimeters','position',[0, 0, plot_width, plot_height])
pause(1)
saveas(gcf, filename_model1_word2vec, save_format)
%% Model 2 --- Perplexity on training set
lines_read = 110

pp1 = dlmread('perplex_1.csv', ';', [0, 0, lines_read, 3]);
pp2 = dlmread('perplex_2.csv', ';', [0, 0, lines_read, 3]);

data_set_size = 500000
batch_1 = 32
batch_2 = 16

steps_1 = pp1(:,1);
steps_2 = pp2(:,1);

percentage_1 = (steps_1*batch_1)/data_set_size;
percentage_2 = (steps_2*batch_2)/data_set_size;

perplex_1 = pp1(:,4);
perplex_2 = pp2(:,4);

model2_perplexity = figure
axes1 = axes('Parent', model2_perplexity);
hold(axes1,'on');

semilogy(percentage_1, perplex_1, '-o', percentage_2, perplex_2, '-x', 'LineWidth', 2)

title('Perplexity for the conditioned model')
xlabel('Percentage of training set')
ylabel('Perplexity')
legend('2*1024 cells', '4*512 cells')
%grid on

if include_origo
    %xlim([0, percentage_1(end)])
    %ylim([0, max(perplex1(1), perplex2(1))])
end

% Fix ticks
box(axes1,'on');
grid(axes1,'on');
% Set the remaining axes properties

x_tics = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
y_tics = [1, 10, 25, 50, 100, 250, 500, 1000, 2500]
x_labels = arrayfun(@(n) strcat(num2str(n), '%') , x_tics, 'unif', 0)
y_labels = arrayfun(@num2str, y_tics, 'unif', 0)
x_tics = x_tics./100

set(axes1,'YMinorTick','on','YScale','log', ...
    'XTick', x_tics, 'XTickLabel', x_labels, ...
    'YTick', y_tics, 'YTickLabel', y_labels);
% Create legend
legend(axes1,'show');

text(percentage_1(end), perplex_1(end), ...
    sprintf('%2.2f', perplex_1(end)), ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'bottom')

text(percentage_2(end), perplex_2(end), ...
    sprintf('%2.2f', perplex_2(end)), ...
    'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'bottom')

ylim([0, max(perplex_1(1), perplex_2(1))])

set(gcf,'units','centimeters','position',[0, 0, plot_width, plot_height])
pause(1)
saveas(gcf, filename_model2_perplexity, save_format)
%axis = gca
%axis.YTick = [1200, 600, 300, 150]

%% Create script to crop the generated PDFs
script_file = fopen('crop_pdfs.sh','w');
fprintf(script_file, '#!/bin/bash\n');
for file = files
    filename = char(strcat(file, '.', save_format))
    fprintf(script_file, 'pdfcrop --margins %d %s\n', ...
        crop_margins, filename);
end
fclose(script_file);
fileattrib('crop_pdfs.sh', '+x');