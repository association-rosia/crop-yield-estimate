# 🌾 Digital Green Crop Yield Estimate Challenge

<img src='assets/banner.png'>

Objective of this challenge is to create a machine learning solution to predict the crop yield per acre of rice or wheat
crops in India.

## 🏆 Challenge ranking
The score of the challenge was the RMSE. 
Our solution was the best one with a RMSE equal to 100.36.

The podium:  
🥇 RosIA - 100.36  
🥈 ihar - 100.68  
🥉 belkasanek - 102.43  

## 🛠️ Data processing

### Pre-processing pipeline 

### GReaT augmentation and imputation

## 🏛️ Model architecture

## #️⃣ Command lines

### Launch training  
```bash
python src/models/train_model.py --estimator_name <estimator_name> --task <task> --nb_agents <nb_agents>
```

### Launch inference
```bash
python src/models/predict_model.py --ensemble_strategy <ensemble_strategy> --class_id <class_id_1> <class_id_2> <class_id_3> --low_id <low_id_1> <low_id_2> <low_id_3> --medium_id <medium_id_1> <medium_id_2> <medium_id_3> --high_id <high_id_1> <high_id_2> <high_id_3>
```
## 📝 Citing

## 🛡️ License

Project is distributed under [MIT License](https://github.com/association-rosia/flair-2/blob/main/LICENSE)

## 👨🏻‍💻 Contributors <a name="contributors"></a>

Louis
REBERGA <a href="https://twitter.com/rbrgAlou"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/louisreberga/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="louis.reberga@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a>

Baptiste
URGELL <a href="https://twitter.com/Baptiste2108"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/baptiste-urgell/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="baptiste.u@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a> 


