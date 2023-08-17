import os
import time
import json
import asyncio
import aiofiles
import pickle

from datetime import datetime
import pandas as pd

from typing import Protocol
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from Agent import Agent
from utils.timeit import timeit, async_timeit
from log.color import LogColor
from utils.file import file_control, get_last_character_from_file
from utils.spinner import Spinner

log = LogColor()

class Slave(Agent):
    def __init__(
        self,
        id: str,
        IP: str,
        port: int | None,
        stream_name: str,
        group_name: str,
        model: BaseEstimator,
        model_params: dict,
    ) -> None:
        super().__init__(id, IP, port, stream_name, group_name)
        self.model = model
        self.model_train= None
        self.model_params = model_params
        self.model_trained = False
        self.model_trained_time = None
        self.model_trained_time_str = None
        self.X_train = None
        self.y_train = None

    @property
    def reading_done(self):
        return self._is_train_reading_done

    @reading_done.setter
    def reading_done(self, _is_train_reading_done):
        self._is_train_reading_done = _is_train_reading_done
        self.train()

    def train_callback(self, future):
        print(f"Model trained at {self.model_trained_time_str}")
    
    async def load_dataset(self):
        X = None
        y = None
        with open(f"output/{self.id}_xunique_train.json", "r") as f:
            X = json.load(f)
        with open(f"output/{self.id}_yunique_train.json", "r") as f:
            y = json.load(f)

        X_train = pd.DataFrame(X["data"]).drop(columns=["999"])
        format_string = "%d %m %Y %H:%M"
        X_train["0"] = pd.to_datetime(X_train["0"], format=format_string).astype("int")

        y_train = pd.DataFrame(y["data"]).drop(columns=["999"])
        y_train = y_train.to_numpy().ravel()
        return X_train, y_train

    async def train(self):
        pickle_file = f"models/{self.id}_model.pkl"
        X_train, y_train=await self.load_dataset()

        if os.path.exists(pickle_file) or self.model_trained:
            log.p_fail(f"Model {self.model} already trained")
            with open(pickle_file, "rb") as f:
                self.model = pickle.load(f)
                print(self.model)
            log.p_ok(f"Model {self.model} loaded from pickle file")
            await self.predict(X_train, y_train)
            return
        
        try:
            # train model
            self.model.fit(X_train, y_train)
            
            # create models dir if not exists
            if not os.path.exists("models"):
                os.mkdir("models")

            # save model to pickle file
            with open(f"models/{self.id}_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
            log.p_ok(f"Model {self.model} pickled")

            # log
            self.model_trained = True
            self.model_trained_time = time.time()
            self.model_trained_time_str = datetime.fromtimestamp(
                self.model_trained_time
            ).strftime("%d/%m/%Y %H:%M:%S")
            log.p_ok(f"Model {type(self.model).__name__} trained at {self.model_trained_time_str}")

            # TODO: i can make X_train and y_train class member and use them in predict...
            await self.predict(X_train, y_train)
        except Exception as e:
            log.p_fail(e)
            log.p_fail(e.__traceback__.tb_lineno)

    async def predict(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        y_pred=self.model.predict(x_train)
        log.p_ok(f"Model {self.model} predicted")
        # metrics 
        mse = mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        log.p_ok(f"Model {self.model} metrics: mse: {mse}, mae: {mae}, r2: {r2}")
        # write metrics to file
        with open(f"models/{self.id}_metrics.txt", "w") as f:
            f.write(f"{self.model_trained_time_str}\n")
            f.write(f"{self.model} mse: {mse}, mae: {mae}, r2: {r2}\n")



    async def read(self):
        no_data_timer = 0
        while True:
            file_name = f"output/{self.id}_xunique_train.json"
            try:
                response = await self.r.xreadgroup(
                    self.group_name, self.stream_name, {self.stream_name: ">"}, None
                )
                if response:
                    for stream_name, stream_data in response:
                        for message_id, message_data in stream_data:
                            decoded_dict = {
                                key.decode(): value.decode()
                                for key, value in message_data.items()
                            }
                            last_col  = decoded_dict["999"]           
                            if last_col == "END":
                                break
                            flag_val = int(float(last_col))
                            if (
                                flag_val == int(self.id[-1])
                                or flag_val == 0
                                or flag_val == int(self.id[-1]) * 10
                                or flag_val == int(79)
                            ):
                                # TODO: if shape == Ytrain, ignore date to float conversion
                                if flag_val == 0:
                                    file_name = f"output/{self.id}_xcommon.json"
                                elif flag_val == int(79):
                                    file_name = f"output/{self.id}_ycommon.json"
                                elif flag_val == int(self.id[-1]) * 10:
                                    file_name = f"output/{self.id}_yunique_train.json"
                                json_string = json.dumps(decoded_dict)
                                # write to file using aiofiles
                                async with aiofiles.open(file_name, mode="a") as f:
                                    # if file is empty, write the first line as [
                                    if os.stat(file_name).st_size == 0:
                                        await f.write('{"data":[\n')
                                    file_control(file_name, self.file_opened)
                                    self.file_opened = True
                                    await f.write(json_string + ",\n")
                                await self.r.xack(
                                    stream_name, self.group_name, message_id
                                )
                else:
                    log.p_fail(f"{log.p_bold(self.id)} No data to read")
                    await asyncio.sleep(1)
                    no_data_timer += 1
                    if no_data_timer == 3:
                        log.p_fail(
                            f"{log.p_bold(self.id)} No data to read for 10 seconds"
                        )
                        # list all files in output dir
                        output_files = os.listdir("output")
                        # add prefix output/
                        output_files = ["output/" + i for i in output_files]
                        for j in output_files:
                            with open(j, mode="a") as f:
                                if get_last_character_from_file(j) != "}":
                                    remove_last_comma_from_file(j)
                                    f.write("\n]}")
                        return

            except Exception as e:
                log.p_fail(
                    f"{log.p_bold(self.id)} Redis Read Exception: {e.with_traceback(e.__traceback__)}{e.__cause__}"
                )
                await asyncio.sleep(1)

    @async_timeit
    async def slave_main(self):
        await self.connect_to_redis()
        await self.create_consumer_group()
        await self.read()

        task = asyncio.create_task(self.train())
        done1, pending1 = await asyncio.wait(task, return_when=asyncio.ALL_COMPLETED)


def remove_last_comma_from_file(file_path: str):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Check if the last line contains a comma
    if lines and lines[-1].strip().endswith(","):
        lines[-1] = lines[-1].strip()[:-1]  # Remove the last comma

    with open(file_path, "w") as f:
        f.writelines(lines)
