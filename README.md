# Feedback loop Simulator


This repo contains the code related to the user simulation model, bootstrapping and feedback loop.

### Setting up

First, create a virtual enviromnent using conda:

```
conda env create -f environment.yml
conda activate <env_name>
```


Then, Clone the repos that implement calibrated Recommendations and the Recommendation model:

1. git clone <git@github.com:caiotda/bpr-mf.git>
2. git clone <git@github.com:caiotda/calibrated-recommendations.git>

For each of the repos cloned, run `pip install -e <repo_name>`. This will improve relative path importing
between projects. The three projects share the same env, so no need to rerun it for every repo.

### Running

1. Download the datasets: simply run `load-datasets --data ml --size s` and `load-datasets --data yelp --size s` for the full preprocessed datasets. This step is implemented in the script `dynamicTasteDistortion/dynamicTasteDistortion/dataset_loader.py`;
2. Run model selection: this steps generate the oracle model and fills up the rating matrix for the associated user sample: `preference-model --data ml --size s --num_users=1000` and `preference-model --data yelp --size s --num_users=1000`. This step also calculates the median time between consecutive interactions for users. We implement this step in `dynamicTasteDistortion/dynamicTasteDistortion/preference_model.py`

3. Run bootstrapping: Bootstrap simulated users preferences until we have around 500k positive interactions (this is code-set currently, but in future versions it could be set via CLI): `bootstrap-preferences --data yelp --size s --num_users=1000` and `bootstrap-preferences --data ml --size s --num_users=1000`. This step is implemented in `dynamicTasteDistortion/dynamicTasteDistortion/bootstrap_preferences.py`


4. Run Feedback loop simulation: In this step, we take the oracle matrix generated in step 2, Hyperparameter tune a BPR model over the bootstrapped set generated in step 3 and start the simulation. Each experiment ran in the paper is defined by a .yaml file in the `dynamicTasteDistortion/dynamicTasteDistortion/experiments/` folder.

    > The experiments related to varying alphas (RQ2) can be found in `dynamicTasteDistortion/dynamicTasteDistortion/experiments/evolving_preferences_movielens/` and  `dynamicTasteDistortion/dynamicTasteDistortion/experiments/evolving_preferences_movielens/`

    > The other experiments can be found in `experiments/ml` and `experiments/yelp`.

To run an experiment, simply run the command `simulate-recommendation --exp_file <exp_file_path.yaml>`

Each experiment file has the following arguments
* size: dataset size: s/m/l; Must match the one used in the previous 2 steps
* data: ml (movielens) or yelp
* num_users: number of users to simulate the recommendation. Also must match parameter used in previous two steps
* model: model used in the recommendation. We currently support unbiased-bpr, bpr, random recommendation, most popular recommendation. In the paper, we report only BPR based results for simplciity
* prefenrece_update_rate: the $\alpha$ parameter. Defaults to 0.1 in our experiments, except for RQ2 related experiments.
* exp_name: name of the experiment. Important only for persistency reasons.

## Experiment data

As of this version of the paper, we only save the .pkl version of the metrics measured at each iteraction. For reproductibility, we compile the metrics related to each plot/table under `dynamicTasteDistortion/dynamicTasteDistortion/reproductibility_data/` in separate csv files that match the numbering of tables and plots (e.g: table1.csv, figure2.csv etc).