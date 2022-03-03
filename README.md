# [Invariant Causal Imitation Learning for Generalizable Policies](https://papers.nips.cc/paper/2021/file/204904e461002b28511d5880e1c36a0f-Paper.pdf)

### Ioana Bica, Daniel Jarrett, Mihaela van der Schaar

#### Neural Information Processing Systems (NeurIPS) 2021



## Dependencies

The code was implemented in Python 3.6 and the following packages are needed for running it:
- gym==0.17.2
- numpy==1.18.2
- pandas==1.0.4
- PyYAML==5.4.1
- tensorflow==1.15.0
- torch==1.6.0
- tqdm==4.32.1
- scipy==1.1.0
- scikit-learn==0.22.2
- stable-baselines==2.10.1

A [`requirements.txt`](./requirements.txt) with all the dependencies is also provided.



## Downloading agents and tasks

The RL agents and tasks can be downloaded from [here](https://drive.google.com/drive/folders/1adHqiXHikltbMojY41VJBe_WFDtQW2ku?usp=sharing). The folder structure at the shared Google Drive follows that of this repository, please place the files into corresponding directories:
```
.
├── contrib
│   └── baseline_zoo
│       └── trained_agents
|           ├── dqn/*
|           └── ppo2/*
└── volume
    └── CartPole-v1/*
```



## Running and evaluating the model:

### OpenAI Gym

The control tasks used for experiments are from OpenAI gym [1]. Each control task is associated with a true reward 
function (unknown to the imitation algorithm). In each case, the “expert” demonstrator can be obtained by using a 
pre-trained and hyperparameter-optimized agent from the RL Baselines Zoo [2] in Stable OpenAI Baselines [3]. 

In this implementation we provide the expert demonstrations for 2 environments for CartPole-v1 in 'volume/CartPole-v1'. Note that the 
code in 'contrib/baselines_zoo' was taken from [2]. 
  
To train and evaluate ICIL on CartPole-v1, run the following command with the chosen command line arguments. For reference, 
the expert performance is 500.

```bash
python testing/il.py
```
```
Options :
   --env                  # Environment name. 
   --num_trajectories	  # Number of expert trajectories used for training the imitation learning algorithm. 
   --trial                # Trial number.
```

Outputs:
   - Average reward for 10 repetitions of running ICIL.  

#### Example usage

```
python testing/il.py  --env='CartPole-v1' --num_trajectories=20 --trial=0 
```

#### References

[1] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang,
and Wojciech Zaremba. Openai gym. OpenAI, 2016

[2] Antonin Raffin. Rl baselines zoo. https://github.com/araffin/rl-baselines-zoo, 2018

[3] Ashley Hill, Antonin Raffin, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, Rene Traore, Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plappert,
Alec Radford, John Schulman, Szymon Sidor, and Yuhuai Wu. Stable baselines. https://github.com/hill-a/stable-baselines, 2018.

### MIMIC-III

In order to run the experiments on the MIMIC-III dataset, you will need to get [MIMIC-III access credentials](https://mimic.mit.edu/docs/gettingstarted/). The preprocessed files can be obtained by contacting the authors (you must confirm your MIMIC access credentials). Place the files under `volume/MIMIC`.

#### Example usage

```
python testing/il_mimic.py --trial=0
``` 



### Citation

If you use this code, please cite:

```
@inproceedings{bica2021invariant,
  title={Invariant Causal Imitation Learning for Generalizable Policies},
  author={Bica, Ioana and Jarrett, Daniel and van der Schaar, Mihaela},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
