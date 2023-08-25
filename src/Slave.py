import os
import time
import json
import asyncio
import aiofiles
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from utils.timeit import async_timeit

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from keras import datasets, layers, models
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

from Agent import Agent
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
        self.X_unique               = None
        self.y_unique               = None
        self.X_common               = None
        self.y_common               = None
        self.X_test                 = None
        self.y_test                 = None 
        self.X_val                  = None
        self.y_val                  = None
        self.X_train                = None
        self.y_train                = None

        self.model                  = model
        self.model_train            = None
        self.model_trained          = False
        self.model_trained_time     = None
        self.model_trained_time_str = None
        self.model_params           = model_params

    
    async def load_dataset(self):
        X = None
        y = None
        X_test = None
        y_test = None
        X_common = None
        y_common = None
        X_val = None
        y_val = None

        try:
            with open(f"output/{self.id}_xunique_train.json", "r") as f:
                X = json.load(f)
            with open(f"output/{self.id}_yunique_train.json", "r") as f:
                y = json.load(f)
            with open(f"output/{self.id}_xtest.json", "r") as f:
                X_test = json.load(f)
            with open(f"output/{self.id}_ytest.json", "r") as f:
                y_test = json.load(f)
            with open(f"output/{self.id}_xcommon.json", "r") as f:
                X_common = json.load(f)
            with open(f"output/{self.id}_ycommon.json", "r") as f:
                y_common = json.load(f)
            with open(f"output/{self.id}_xval.json", "r") as f:
                X_val = json.load(f)
            with open(f"output/{self.id}_yval.json", "r") as f:
                y_val = json.load(f)
        except Exception as e:
            log.p_fail(f"Read json failed: {e}")
            log.p_fail(e.__traceback__.tb_lineno)

        try:
            format_string = "%d %m %Y %H:%M"
            self.X_unique = pd.DataFrame(X["data"]).drop(columns=["999"])
            self.X_unique["0"] = pd.to_datetime(self.X_unique["0"], format=format_string).astype("int")

            self.X_test = pd.DataFrame(X_test["data"]).drop(columns=["999"])
            self.X_test["0"] = pd.to_datetime(self.X_test["0"], format=format_string).astype("int")

            self.y_test = pd.DataFrame(y_test["data"]).drop(columns=["999"])
            self.y_test = self.y_test.to_numpy().ravel()

            self.X_common = pd.DataFrame(X_common["data"]).drop(columns=["999"])
            self.X_common["0"] = pd.to_datetime(self.X_common["0"], format=format_string).astype("int")

            self.y_unique = pd.DataFrame(y["data"]).drop(columns=["999"])
            self.y_common = pd.DataFrame(y_common["data"]).drop(columns=["999"])

            self.X_val = pd.DataFrame(X_val["data"]).drop(columns=["999"])
            self.X_val["0"] = pd.to_datetime(self.X_val["0"], format=format_string).astype("int")

            self.y_val = pd.DataFrame(y_val["data"]).drop(columns=["999"])
            self.y_val = self.y_val.to_numpy().ravel()

            '''
            Combine unique and common into train dataset
            '''
            self.X_train = pd.concat([self.X_unique, self.X_common])
            self.y_train = pd.concat([pd.DataFrame(self.y_unique), pd.DataFrame(self.y_common)])
            print(self.X_train.shape, self.y_train.shape)
            self.y_train = self.y_train.to_numpy().ravel()
        except Exception as e:
            log.p_fail(f"Load dataset failed: {e}")
            log.p_fail(e.__traceback__.tb_lineno)


    async def train(self):
        pickle_file = f"models/{self.id}_model.pkl"

        if os.path.exists(pickle_file) or self.model_trained:
            log.p_fail(f"Model {self.model} already trained")
            with open(pickle_file, "rb") as f:
                self.model = pickle.load(f)
            log.p_ok(f"Model {self.model} loaded from pickle file")
            await self.predict()
            return
        
        try:
            # train model
            self.model.fit(self.X_train, self.y_train)
            
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

            await self.predict()
        except Exception as e:
            log.p_fail(e)
            log.p_fail(e.__traceback__.tb_lineno)

    async def CNN(self):
        try:
            # change datetime to float (int is not enough)
            self.X_train["0"] = self.X_train["0"].to_numpy().astype(np.float32)
            self.X_val["0"]   = self.X_val  ["0"].to_numpy().astype(np.float32)
            self.X_test["0"]  = self.X_test ["0"].to_numpy().astype(np.float32)
            
            input_dim = self.X_train.shape[1]
            model = Sequential([
                Dense(64, activation='relu', input_shape=(input_dim,)),
                Dense(32, activation='relu'),
                Dense(1)  # Output layer for power prediction
            ])
            model.compile(loss='mean_squared_error', optimizer='adam')
            log.p_ok(f"Model {model} compiled")
            model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, validation_data=(self.X_val, self.y_val))
            #loss = model.evaluate(X_test, y_test)
            model.save(f'models/{self.id}_cnn.h5')
        except Exception as e:
            log.p_fail(e)
            log.p_fail(e.__traceback__.tb_lineno)

    async def predict(self):
        y_pred=self.model.predict(self.X_test)
        log.p_ok(f"Model {self.model} predicted")
        # metrics 
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
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
                                or flag_val == -1
                                or flag_val == -2
                                or flag_val == -3
                                or flag_val == -4
                            ):
                                # TODO: if shape == Ytrain, ignore date to float conversion
                                if flag_val == 0:
                                    file_name = f"output/{self.id}_xcommon.json"
                                elif flag_val == int(79):
                                    file_name = f"output/{self.id}_ycommon.json"
                                elif flag_val == int(self.id[-1]) * 10:
                                    file_name = f"output/{self.id}_yunique_train.json"
                                elif flag_val == -1:
                                    file_name = f"output/{self.id}_xtest.json"
                                elif flag_val == -2:
                                    file_name = f"output/{self.id}_ytest.json"
                                elif flag_val == -3:
                                    file_name = f"output/{self.id}_xval.json"
                                elif flag_val == -4:
                                    file_name = f"output/{self.id}_yval.json"

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
        await self.load_dataset()
        
        task = asyncio.create_task(self.train())
        done1, pending1 = await asyncio.wait(task, return_when=asyncio.ALL_COMPLETED)

        #task_cnn = asyncio.create_task(self.CNN())
        #done2, pending2 = await asyncio.wait(task_cnn, return_when=asyncio.ALL_COMPLETED)

        if self.model.__name__ == "CNN":
            pass
            

def remove_last_comma_from_file(file_path: str):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Check if the last line contains a comma
    if lines and lines[-1].strip().endswith(","):
        lines[-1] = lines[-1].strip()[:-1]  # Remove the last comma

    with open(file_path, "w") as f:
        f.writelines(lines)