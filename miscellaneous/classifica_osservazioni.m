function  pred_labels = classifica_osservazioni(obs_features, soglie)

%   Funzione per simulare la fase di predizione di un classificatore.
%
%   Es.: >> predicted_labels = classifica_osservazione(test_fetures, training_params);
%   
%   N.B.: "training_params" è l'output della funzione "addestra_classificatore()".


    pred_labels = [];
    for i = 1:size(obs_features, 1)
        
        obs_i_features = obs_features(i, :);
        
        % Calola distanza osservazione da centri delle classi
        dist_obs_i_c1 = sqrt(sum((obs_i_features - soglie.centro_1).^2, 2));
        dist_obs_i_c2 = sqrt(sum((obs_i_features - soglie.centro_2).^2, 2));
        dist_obs_i_c3 = sqrt(sum((obs_i_features - soglie.centro_3).^2, 2));
        dist_min = min([dist_obs_i_c1 dist_obs_i_c2 dist_obs_i_c3]);
        
        % Classifica in base alla distanza minima
        if dist_min == dist_obs_i_c1
            pred_i_class = 1;
        elseif dist_min == dist_obs_i_c2
            pred_i_class = 2;
        elseif dist_min == dist_obs_i_c3
            pred_i_class = 3;
        end
        
        pred_labels = cat(1, pred_labels, pred_i_class);
    end

end