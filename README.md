# BeerGAME_DQN
Course Lab1 of PKU: Multi-Agent system(24-Spring). An DQN agent for the classical BeerGAME. 
This project is based on the BeerGAME environment: [BeerGAME](https://github.com/YanSong97/BeerGame_demo)


![f6141d8e71ee39e704d5ccee4655e80](https://github.com/spidermonk7/BeerGAME_DQN/assets/98212025/7b7c7278-a715-4a38-ab0a-6b144587343a)

## Quickstart
To train a DQN model with other players using bs strategy, please run:
  
      python new_env.py --mode=train --competitors=bs


The trained model will be saved in model/MODEL_NAME.pth.To test their performance on 10 test games, you can run:
      
      python new_env.py --mode=test --competitors=DQN --model=YOUR_MODEL_NAME

Find the models in folder models/.. (e.g. --model=Q_net64_5_prF.pth). After a while the programme will return an figure shows the rewards.
![demo_valid](https://github.com/spidermonk7/BeerGAME_DQN/assets/98212025/60c824c2-acdb-4b6f-80c4-e74bc7987f80)

## Arguments:
Here's the description of optional arguments:
| arg | default | -help | 
| :-----:| :----: | :----: |
| --mode | train | train or test |
| --model | Q_net64_5_prF.pth | model name |
| --k | 5 | size of minibatch |
| --priority | False | if use priority replay |
| --competitors | DQN | strategy of other players(in bs, random, DQN) |

And for detailed discussions, please check my report: **MA__BeerGame (2).pdf**


    
