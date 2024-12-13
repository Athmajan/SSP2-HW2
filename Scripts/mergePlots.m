% Open the figures
fig_caseI = openfig('channel_realization_caseI.fig');
fig_caseII = openfig('channel_realization_caseII.fig');
fig_caseIII = openfig('channel_realization_caseIII.fig');

% Extract axes and children from each figure
ax_caseI = findobj(fig_caseI, 'type', 'axes');
ax_caseII = findobj(fig_caseII, 'type', 'axes');
ax_caseIII = findobj(fig_caseIII, 'type', 'axes');

% Create a new figure and plot the contents of each figure
figure;
hold on;

% Plot from the first figure
for i = 1:length(ax_caseI.Children)
    copyobj(ax_caseI.Children(i), gca);
end

% Plot from the second figure
for i = 1:length(ax_caseII.Children)
    copyobj(ax_caseII.Children(i), gca);
end

% Plot from the third figure
for i = 1:length(ax_caseIII.Children)
    copyobj(ax_caseIII.Children(i), gca);
end

hold off;

% Add labels and legend
xlabel('Sample number, n');
ylabel('Tap weight, h[n]');
legend('show');

% Close the opened figures if needed
close(fig_caseI);
close(fig_caseII);
close(fig_caseIII);
