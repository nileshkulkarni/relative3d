function [gmm] = fitGMM(X, K)
% 	gmm = fitgmdist(X(:,1:2), K, 'RegularizationValue', 1E-3);
	gmm = fitgmdist(X(:,1:3), K, 'RegularizationValue', 1E-3);
	
    max_d = max(X, [], 1);
	min_d = min(X, [], 1);
	[X,Y,Z] = meshgrid(min_d(1):0.1:max_d(1),min_d(2):0.1:max_d(2),min_d(3):0.1:max_d(3));
	scores = reshape(pdf(gmm,[X(:) Y(:), Z(:)]), size(X,1),size(Y,2), size(Z,3));
    figure;
    %scatHand = scatter3(X(:), Y(:), Z(:));
    clr = ind2rgb(scores(:), jet(256));
    % clr = repmat(scores(:),1,3);
    set(scatHand, 'CData', clr);
%     surf(X,Y,Z,scores)
%     colorbar
%     shading interp
    % 	[X,Y] = meshgrid(min_d(1):0.1:max_d(1),min_d(2):0.1:max_d(2));
%     contour(X,Y,reshape(pdf(gmm,[X(:) Y(:)]),size(X,1),size(Y,2)),20)
end