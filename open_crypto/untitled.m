% plot_accuracy_overview.m
% Erstellt ein Balkendiagramm aus allen Accuracy-Werten in Excel-Dateien
% Erwartet in jedem Excel ein Sheet "metrics" mit Spalten:
% metric | value
% und einer Zeile mit metric == 'accuracy'

clear;
clc;

fprintf("=== Accuracy Overview Plotter (MATLAB) ===\n");

% Ordner mit den Excel-Dateien (aktuelles Skript-Verzeichnis)
baseDir = fileparts(mfilename('fullpath'));
cd(baseDir);

files = dir("*.xlsx");

if isempty(files)
    error("Keine .xlsx Dateien im Ordner gefunden.");
end

names = {};
acc_values = [];

for i = 1:length(files)
    fname = files(i).name;

    try
        T = readtable(fname, "Sheet", "metrics");
    catch
        fprintf("Überspringe %s (kein Sheet 'metrics')\n", fname);
        continue;
    end

    % Spaltennamen normalisieren
    vars = lower(strtrim(T.Properties.VariableNames));
    T.Properties.VariableNames = vars;

    if ~ismember("metric", vars) || ~ismember("value", vars)
        fprintf("Überspringe %s (falsche Spalten)\n", fname);
        continue;
    end

    % metric == 'accuracy' suchen (case-insensitive)
    m = lower(string(T.metric));
    idx = find(m == "accuracy", 1);

    if isempty(idx)
        fprintf("Überspringe %s (keine accuracy Zeile)\n", fname);
        continue;
    end

    val = T.value(idx);

    % Falls als Text mit Komma "0,51"
    if iscell(val) || isstring(val)
        s = strrep(string(val), ",", ".");
        val = str2double(s);
    end

    if isnan(val)
        fprintf("Überspringe %s (accuracy nicht numerisch)\n", fname);
        continue;
    end

    names{end+1} = fname; %#ok<SAGROW>
    acc_values(end+1) = val; %#ok<SAGROW>
end

if isempty(acc_values)
    error("Keine gültigen Accuracy-Werte gefunden.");
end

% Tabelle erstellen und nach Accuracy sortieren
T_out = table(names', acc_values', ...
    'VariableNames', {'file', 'accuracy'});

T_out = sortrows(T_out, 'accuracy', 'descend');

disp("Gefundene Accuracy-Werte:");
disp(T_out);

% --------------------------------------------------
% Balkendiagramm
% --------------------------------------------------
figure;
bar(T_out.accuracy);
yline(0.5, '--');   % Zufalls-Baseline

xticks(1:height(T_out));
xticklabels(T_out.file);
xtickangle(75);

ylabel("Accuracy");
title("Übersicht: Accuracy aus allen Backtests");

grid on;
set(gca, "FontSize", 10);

% --------------------------------------------------
% Grafik speichern
% --------------------------------------------------
out_png = fullfile(baseDir, "accuracy_overview.png");
saveas(gcf, out_png);

% Optional: CSV speichern
out_csv = fullfile(baseDir, "accuracy_overview.csv");
writetable(T_out, out_csv);

fprintf("\n✅ Grafik gespeichert: %s\n", out_png);
fprintf("✅ CSV gespeichert   : %s\n", out_csv);
