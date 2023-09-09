# summer-intern
in ini file slave agent's name must end with integer and must me sequential. 
For example: 
agent1, agent2, agent3, agent4 or 
a1, a2, a3, a4 or
xgboost1, adaboost2, lightgbm3, randomforest4

## Development

``` bash
# for lightgbm installation https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#visual-studio-or-vs-build-tools

# after cloning and navigating to the directory
python -m venv venv
source venv/bin/activate # for linux and mac
venv\Scripts\Activate.ps1 # for windows
pip install -r requirements.txt

# you can run main file like below example
# main file only helps with running multiple agents
# you can call master_main or slave_main using asyncio seperately
python src/main.py --slave agent1 agent2 agent3 agent4

```

## TO-DO's
- [ ] Fix slave count --> master
- [ ] Best slave selection
- [ ] Combine unique and common
    - for now training only done with unique, with no common dataset input 
- [ ] xgboost,lightgbm model
- [ ] timing for each agent
- [ ] voting for regression models
- [ ] training unique and common ratio between (0,100,5) 
- [ ] code doesnt exits because of slave listen loop

