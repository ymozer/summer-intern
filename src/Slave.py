from Agent import Agent

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor

class Slave(Agent):
    def __init__(self, id: str, IP: str, port:int|None, stream_name:str, group_name:str) -> None:
         super().__init__(id, IP, port, stream_name, group_name)


    async def main(self):
        await self.connect_to_redis()
        await self.create_consumer_group()
        await self.read()


    
