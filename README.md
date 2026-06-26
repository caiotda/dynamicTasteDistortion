# Feedback loop Simulator


This repo contains the code related to the user simulation model, bootstrapping and feedback loop.

### Setting up

First, create a virtual enviromnent using conda:

```
conda env create -f environment.yml
conda activate <env_name>
```


Then, Clone the repos that implement calibrated Recommendations and the Recommendation model:

1. Clone from https://anonymous.4open.science/r/calibrated-recommendations-D0CE/
2. and from https://anonymous.4open.science/r/bpr-mf-87FA

For each of the repos cloned, run `pip install -e <repo_name>`. This will improve relative path importing
between projects. The three projects share the same env, so no need to rerun it for every repo.

### Running

1. Download the dataset and preprocess: simply run `load-datasets --data globo --size s` for the full preprocessed dataset. This step is implemented in the script `dynamicTasteDistortion/dynamicTasteDistortion/dataset_loader.py`;
2. Run model selection: this steps generate the oracle model and fills up the rating matrix for the associated user sample: `preference-model --data globo --size s --num_users=1000`. This step also calculates the median time between consecutive interactions for users. We implement this step in `dynamicTasteDistortion/dynamicTasteDistortion/preference_model.py`

3. Run bootstrapping: Bootstrap simulated users preferences until we have around 500k positive interactions (this is code-set currently, but in future versions it could be set via CLI): `bootstrap-preferences --data globo --size s --num_users=1000`. This step is implemented in `dynamicTasteDistortion/dynamicTasteDistortion/bootstrap_preferences.py`


4. Run Feedback loop simulation: In this step, we take the oracle matrix generated in step 2, Hyperparameter tune a BPR model over the bootstrapped set generated in step 3 and start the simulation. Each experiment ran in the paper is defined by a .yaml file in the `dynamicTasteDistortion/dynamicTasteDistortion/experiments/globo` folder.
    

To run an experiment, simply run the command `simulate-recommendation --exp_file <exp_file_path.yaml>`

Each experiment file has the following arguments
* size: dataset size: s/m/l; Must match the one used in the previous 2 steps
* data: globo. Our code works for movielens and yelp, but the results reported are exclusively to globo.com
* num_users: number of users to simulate the recommendation. Also must match parameter used in previous two steps
* model: model used in the recommendation. We currently support unbiased-bpr, bpr, random recommendation, most popular recommendation. In the paper, we report only BPR based results for simplciity
* prefenrece_update_rate: the $\alpha$ parameter. Defaults to 0.1 in our experiments.
* exp_name: name of the experiment. Important only for persistency reasons.
* [OPTIONAL] params: bpr params to be used. This overwrites the hyperparameter tuning flow. To force hyperparameter selection, remove the `params` field from each experiment.


## Experiment data

As of this version of the paper, we only save the .pkl version of the metrics measured at each iteraction. For reproductibility, we compile the metrics related to each plot/table under `dynamicTasteDistortion/dynamicTasteDistortion/reproductibility_data/` in separate csv files that match the numbering of tables and plots (e.g: table1.csv, figure2.csv etc). Refer to https://anonymous.4open.science/r/dynamicTasteDistortion-webmedia/dynamicTasteDistortion/reproductibility_data/ for data pertaining to each figure