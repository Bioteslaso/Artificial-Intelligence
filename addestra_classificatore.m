function parametri = addestra_classificatore(obs_features, obs_labels)

%   Funzione per simulare la fase di addestramento di un classificatore.
%
%   Es.: >> training_params = addestra_classificatore(training_features, training_labels);


    % Raggruppa le osservazioni per classe
    obs_feat_c1 = obs_features(obs_labels == 1, :);
    obs_feat_c2 = obs_features(obs_labels == 2, :);
    obs_feat_c3 = obs_features(obs_labels == 3, :);
    
    % Addestra il classificatore in base alle osservazioni per classe
    parametri.centro_1 = mean(obs_feat_c1, 1);
    parametri.centro_2 = mean(obs_feat_c2, 1);
    parametri.centro_3 = mean(obs_feat_c3, 1);

end