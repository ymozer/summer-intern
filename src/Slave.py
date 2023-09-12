import os
import re
import time
import json
import asyncio
import aiofiles
import pickle
import pandas as pd
import numpy as np

from keras import Sequential
from keras.layers import Dense

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from sklearn.svm import SVR
from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Agent import Agent
from log.color import LogColor
from utils.file import file_control, get_last_character_from_file

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
        no_data_timer: int,
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

        self.mse                    = None
        self.mae                    = None
        self.r2                     = None


        self.model                  = model
        self.model_train            = None
        self.model_trained          = False
        self.model_trained_time     = None
        self.model_trained_time_str = None
        self.model_params           = model_params
        self.no_data_timer          = no_data_timer

    
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
            
            self.X_unique = pd.DataFrame(X["data"]).drop(columns=["999"])
            self.X_test = pd.DataFrame(X_test["data"]).drop(columns=["999"])
            self.y_test = pd.DataFrame(y_test["data"]).drop(columns=["999"])
            self.y_test = self.y_test.to_numpy().ravel()
            self.X_common = pd.DataFrame(X_common["data"]).drop(columns=["999"])
            self.y_unique = pd.DataFrame(y["data"]).drop(columns=["999"])
            self.y_common = pd.DataFrame(y_common["data"]).drop(columns=["999"])

            self.X_val = pd.DataFrame(X_val["data"]).drop(columns=["999"])
            self.y_val = pd.DataFrame(y_val["data"]).drop(columns=["999"])
            self.y_val = self.y_val.to_numpy().ravel()

            '''
            Combine unique and common into train dataset
            '''
            self.X_train = pd.concat([self.X_unique, self.X_common])
            self.y_train = pd.concat([pd.DataFrame(self.y_unique), pd.DataFrame(self.y_common)])
            self.y_train = self.y_train.to_numpy().ravel()
        except Exception as e:
            log.p_fail(f"Load dataset failed: {e}")
            log.p_fail(e.__traceback__.tb_lineno)


    async def train(self):

        start = time.time()
        pickle_file = f"models/{self.id}_model.pkl"

        if os.path.exists(pickle_file) or self.model_trained:
            '''
            If model is already trained, load it from pickle file
            '''
            log.p_fail(f"Model {self.model} already trained")
            with open(pickle_file, "rb") as f:
                self.model = pickle.load(f)
            log.p_ok(f"Model {self.model} loaded from pickle file")

            if str(type(self.model)) == str(type(LGBMRegressor())):
                log.p_warn("LGBMRegressor")
            elif str(type(self.model)) == str(type(XGBRegressor())):
                log.p_warn("XGBRegressor")
            elif str(type(self.model)) == str(type(SVR())):
                log.p_warn("SVR")
            else:
                log.p_warn("Predict default models")
                await self.my_predict()
            return
        
        else:
            log.p_ok(f"Model {self.model} not trained")

            try:
                #self.X_train = SelectKBest(f_regression, k=5).fit_transform(self.X_train, self.y_train)
                #self.X_test = SelectKBest(f_regression, k=5).fit_transform(self.X_test, self.y_test)
                #self.X_val = SelectKBest(f_regression, k=5).fit_transform(self.X_val, self.y_val)
                #self.X_train = SelectFromModel(self.model, prefit=False).fit_transform(self.X_train, self.y_train)
                #self.X_test = SelectFromModel(self.model, prefit=False).fit_transform(self.X_test, self.y_test)
                #self.X_val = SelectFromModel(self.model, prefit=False).fit_transform(self.X_val, self.y_val)

                # train model
                if str(type(self.model)) == str(type(LGBMRegressor()))\
                    or str(type(self.model)) == str(type(XGBRegressor())):
                    
                    self.X_train = self.X_train.to_numpy()
                    self.X_val = self.X_val.to_numpy()
                    self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)])
                    self.model_trained_time = time.time() - start
                    log.p_header(f"Model {self.model} trained with time: {self.model_trained_time:.2f} sec")

                elif str(type(self.model)) == str(type(SVR())):
                    log.p_underline("SVR")
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),  
                        ('svr', self.model) 
                    ])
                    try:
                        pipeline.fit(self.X_train, self.y_train)
                        self.model = pipeline
                        '''
                        # Define a grid of hyperparameters to search
                        param_grid = {
                            'svr__kernel': ['linear', 'rbf', 'poly'],  # Kernels to try
                            'svr__C': [0.1, 1, 10],  # Regularization parameter
                            'svr__gamma': [0.1, 1, 'scale', 'auto'],  # Kernel coefficient (only for 'rbf' and 'poly' kernels)
                            'svr__degree': [2, 3]  # Polynomial degree (only for 'poly' kernel)
                        }

                        # Create a GridSearchCV object
                        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=2, scoring='neg_mean_squared_error',verbose=3)
                        # Fit the grid search to the data
                        grid_search.fit(self.X_train, self.y_train)

                        # Get the best parameters and estimator from the grid search
                        best_params = grid_search.best_params_
                        best_estimator = grid_search.best_estimator_

                        # Print the best hyperparameters
                        log.p_header(f"Best hyperparameters:\n{best_params}\n")

                        # Print the best model
                        log.p_header(f"Best model:\n{best_estimator}\n")
                        self.model=best_estimator
                        '''

                        self.model_trained_time = time.time() - start
                        log.p_header(f"Model {self.model} trained with time: {self.model_trained_time:.2f} sec")
                    except Exception as e:
                        log.p_fail("Pipeline fit failed", e)
                        log.p_fail(e.__traceback__.tb_lineno)
                elif str(type(self.model)) == str(type(Sequential())):
                    try:
                        self.model = Sequential([
                            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
                            Dense(32, activation='relu'),
                            Dense(1)  # Output layer with 1 neuron for regression
                        ])
                        # Compile the model
                        self.model.compile(optimizer='adam', loss='mean_squared_error')
                        history = self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, validation_split=0.2)
                        self.model.save(f'models/{self.id}_cnn.h5')
                        loss = self.model.evaluate(self.X_test, self.y_test)
                        print(f"Test Loss: {loss}")
                    except Exception as e:
                        log.p_fail("Sequential fit failed", e)
                        log.p_fail(e.__traceback__.tb_lineno)
                    pass
                else:
                    log.p_warn("Default models")
                    self.model.fit(self.X_train, self.y_train)

                # save model to pickle file
                with open(f"models/{self.id}_model.pkl", "wb") as f:
                    pickle.dump(self.model, f)
                log.p_ok(f"Model {self.model} pickled")

                await self.my_predict()

            except Exception as e:
                log.p_fail(e)
                log.p_fail(e.__traceback__.tb_lineno)

    async def my_predict(self):
        if str(type(self.model)) == str(type(LGBMRegressor())) or\
            str(type(self.model)) == str(type(XGBRegressor())):
            self.X_test = self.X_test.to_numpy()
            try:
                y_pred = self.model.predict(self.X_test)
            except Exception as e:
                log.p_fail("Predict failed", e)
                log.p_fail(f"{e.__traceback__.tb_lineno}")

        else:
            y_pred=self.model.predict(self.X_test)
        
        log.p_ok(f"Model {self.model} predicted")
        # metrics 
        self.mse = mean_squared_error(self.y_test, y_pred)
        self.mae = mean_absolute_error(self.y_test, y_pred)
        self.r2 = r2_score(self.y_test, y_pred)
        log.p_ok(f"Model {self.model} metrics: mse: {self.mse}, mae: {self.mae}, r2: {self.r2}")
        try:
            # write metrics to file
            with open(f"models/{self.id}_metrics.txt", "w") as f:
                f.write(f"{self.model_trained_time} second\n")
                f.write(f"{str(self.model)}\n mse: {self.mse}, mae: {self.mae}, r2: {self.r2}\n")

            # save predictions and y test to file side by side
            with open(f"models/{self.id}_predictions.csv", "w") as f:
                f.write("y_pred;y_test\n")
                for i in range(len(y_pred)):
                    f.write(f"{y_pred[i]:.2f};{self.y_test[i]}\n")
        except Exception as e:
            log.p_fail("Write metrics and pred error ",e)


        

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

    async def read(self):
        timer = 0
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
                            letterless_id = re.sub(r'[a-zA-Z]', '', self.id)[0]
                            flag_val = int(float(last_col))
                            if (
                                flag_val    == int(letterless_id)      # train x unique
                                or flag_val == int(letterless_id) * 10 # train y unique
                                or flag_val == 0                    # common x
                                or flag_val == int(79)              # common y    
                                or flag_val == -1                   # test x
                                or flag_val == -2                   # test y
                                or flag_val == -3                   # val x
                                or flag_val == -4                   # val y
                            ):
                                # TODO: if shape == Ytrain, ignore date to float conversion
                                if flag_val == 0:
                                    file_name = f"output/{self.id}_xcommon.json"
                                elif flag_val == int(79):
                                    file_name = f"output/{self.id}_ycommon.json"
                                elif flag_val == int(letterless_id) * 10:
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
                    timer += 1
                    if timer > self.no_data_timer:
                        log.p_fail(
                            f"{log.p_bold(self.id)} No data to read for {self.no_data_timer} seconds"
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
                log.p_fail(e.__traceback__.tb_lineno)
                await asyncio.sleep(1)

    async def slave_main(self):
        await self.connect_to_redis()
        await self.create_consumer_group()
        await self.read()
        await self.load_dataset()
        await self.train()
         
        data = {
            "id": self.id,
            "model": str(self.model),
            "model_trained_time": self.model_trained_time,
            "model_trained_time_str": self.model_trained_time_str,
            "mse": self.mse,
            "mae": self.mae,
            "r2": self.r2,
        }
        
        try:
            await self.r.xadd("stream_1", data, "*")
        except Exception as e:
            log.p_fail(f"Write failed", e)
            log.p_fail(e.__traceback__.tb_lineno)
            

def remove_last_comma_from_file(file_path: str):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Check if the last line contains a comma
    if lines and lines[-1].strip().endswith(","):
        lines[-1] = lines[-1].strip()[:-1]  # Remove the last comma

    with open(file_path, "w") as f:
        f.writelines(lines)