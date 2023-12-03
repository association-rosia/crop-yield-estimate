#python src/models/train_model.py --estimator_name LightGBM --task classification --nb_agents 5
#python src/models/train_model.py --estimator_name LightGBM --task reg_l --nb_agents 5
#python src/models/train_model.py --estimator_name LightGBM --task reg_m --nb_agents 5
#python src/models/train_model.py --estimator_name LightGBM --task reg_h --nb_agents 5
python src/models/train_model.py --estimator_name CatBoost --task reg_m --nb_agents 6
python src/models/train_model.py --estimator_name CatBoost --task classification --nb_agents 6