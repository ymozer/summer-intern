# summer-intern
## Development

``` bash
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

