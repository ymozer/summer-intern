from Agent import Agent
from utils.timeit import timeit, async_timeit
import asyncio
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)


class Slave(Agent):
    def __init__(
        self, id: str, IP: str, port: int | None, stream_name: str, group_name: str
    ) -> None:
        super().__init__(id, IP, port, stream_name, group_name)
        self.model = None
        self.model_name = None
        self.model_type = None
        self.model_params = None
        self.model_trained = False
        self.model_trained_time = None
        self.model_trained_time_str = None
        self.X_train = None
        self.y_train = None

    @async_timeit
    async def train(self, model_type, model_params):
        self.model_type = model_type
        self.model_params = model_params
        if model_type == "LinearRegression":
            self.model = LinearRegression(**model_params)
        elif model_type == "DecisionTreeRegressor":
            self.model = DecisionTreeRegressor(**model_params)
        elif model_type == "ExtraTreesRegressor":
            self.model = ExtraTreesRegressor(**model_params)
        elif model_type == "AdaBoostRegressor":
            self.model = AdaBoostRegressor(**model_params)
        elif model_type == "RandomForestRegressor":
            self.model = RandomForestRegressor(**model_params)
        else:
            raise Exception("Invalid model type")

        X_train = Agent.read_json(f"output/{self.id}_xunique_train.json")
        y_train = Agent.read_json(f"output/{self.id}_yunique_train.json")
        self.model.fit(self.X_train, self.y_train)

    @async_timeit
    async def slave_main(self):
        await self.connect_to_redis()
        await self.create_consumer_group()
        await self.read()
        await asyncio.sleep(1)
