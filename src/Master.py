from Agent import Agent
from log.color import LogColor
from utils.timeit import timeit, async_timeit

import os
import sys
import time
import json
import random
import datetime
import asyncio
import pandas as pd
import numpy as np
from asyncio import sleep
from sklearn.model_selection import train_test_split


log = LogColor()


class Master(Agent):
    def __init__(
        self,
        id: str,
        IP: str,
        port: int | None,
        stream_name: str,
        group_name: str,
        delay: float,
        dataset_path: str,
        common_ratio: float,
        test_ratio: float,
        validation_ratio: float,
        y_column_name: str,
        slave_count: int,
    ) -> None:
        super().__init__(id, IP, port, stream_name, group_name)
        self.dataset_path = dataset_path
        self.unique_data_sent = False
        self.delay = delay
        self.common_train_data_sent = False
        self.common_ratio: float = common_ratio
        self.test_ratio: float = test_ratio
        self.validation_ratio: float = validation_ratio
        self.Y_column_name = y_column_name,
        self.slave_count = slave_count


    async def slave_counter(self):
        info = await self.r.execute_command("XINFO", "GROUPS", self.stream_name)

        dictt = self.decode_list_of_bytes(info)
        json_data = json.dumps(dictt, indent=4)
        # pretty print
        log.p_ok(f"data: {json_data}")

        # check if consumers are bigger than 0
        counter = 0
        for i in dictt:
            if i["consumers"] > 0:
                counter += 1
        self.slave_count = len(dictt) - 1

        log.p_ok(f"slave count: {self.slave_count}")
        # create a file for keeping log on is this file executed before
    def decode_list_of_bytes(self, nested_list):
        decoded_list = [
            [item.decode() if isinstance(item, bytes) else item for item in sublist]
            for sublist in nested_list
        ]
        # create a dictionary from even-indexed and odd-indexed pairs for each sublist
        decoded_dict = [dict(zip(even[::2], even[1::2])) for even in decoded_list]
        return decoded_dict

    def split(
        self,
        df,
        number_of_slaves: int = 2,
        Y_column_name: str = "LV ActivePower (kW)",
        random_state: int = 42,
        shuffle: bool = False,
    ):
        """
        Split the data into training and testing sets
        """
        if type(Y_column_name) == tuple:
            Y_column_name = Y_column_name[0]

        X = df.drop(columns=Y_column_name, axis=1)
        y = df[Y_column_name]
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_ratio,
            random_state=random_state, shuffle=shuffle
        )


        # split training data into training and validation
        # 0.25 x 0.8 = 0.2
        (x_train, x_val, y_train, y_val) = train_test_split(
            x_train, y_train, test_size=self.validation_ratio, 
            random_state=random_state, shuffle=shuffle
        ) 

        # split training again for unique and common
        (x_train_unique, x_train_common,
          y_train_unique,y_train_common) = train_test_split(
            x_train, y_train, test_size=self.common_ratio,
            random_state=random_state, shuffle=shuffle
        )

        # split again for each number of slaves
        x_train_unique_split = np.array_split(x_train_unique, number_of_slaves)
        y_train_unique_split = np.array_split(y_train_unique, number_of_slaves)

        log.p_ok(
            f"{log.p_bold(self.id)} Unique data split into {len(x_train_unique_split)} parts"
        )

        for i in range(number_of_slaves):
            """Add identifier column for slaves"""
            num_rows_x = x_train_unique_split[i].shape[0]
            new_column_x = np.full((num_rows_x, 1), i + 1)

            num_rows_y = y_train_unique_split[i].shape[0]
            new_column_y = np.full((num_rows_y, 1), (i + 1) * 10)

            x_train_unique_split[i] = np.hstack((x_train_unique_split[i], new_column_x))
            x_train_unique_split[i] = pd.DataFrame(x_train_unique_split[i])
            # get last column name xtrainuniquesplit

            x_train_last_column_name = x_train_unique_split[i].columns[-1]
            x_train_unique_split[i].rename(columns={x_train_last_column_name: "999"}, inplace=True)
            x_train_unique_split[i]["999"] = new_column_x

            y_train_unique_split[i] = pd.DataFrame(y_train_unique_split[i])
            # rename first column to 0
            y_train_unique_split[i].columns = range(len(y_train_unique_split[i].columns))
            y_train_unique_split[i]["999"] = new_column_y

        # convert columns to range values (0,1,2..)
        y_train_common = pd.DataFrame(y_train_common)
        x_train_common.columns = range(len(x_train_common.columns))
        y_train_common.columns = range(len(y_train_common.columns))
        x_train_common["999"] = 0
        y_train_common["999"] = 79 

        y_test = pd.DataFrame(y_test)
        x_test.columns = range(len(x_test.columns))
        y_test.columns = range(len(y_test.columns))
        x_test["999"]=-1
        y_test["999"]=-2

        y_val = pd.DataFrame(y_val)
        x_val.columns = range(len(x_val.columns))
        y_val.columns = range(len(y_val.columns))
        x_val["999"]=-3
        y_val["999"]=-4
        
        return (
            x_train_unique_split,
            y_train_unique_split,
            x_train_common,
            y_train_common,
            x_test,
            y_test,
            x_val,
            y_val
        )

    async def send(self, dataTsend: pd.DataFrame, agentid: int, *args):
        try:
            for index, row in dataTsend.iterrows():
                await self.write(row.to_dict())
        except Exception as e:
            log.p_fail(f"Redis send exception: {log.p_bold(self.id)} {e}")
        return

    async def send_all(self, splits):
        x_train_unique = splits[0]
        y_train_unique = splits[1]
        x_train_common = splits[2]
        y_train_common = splits[3]
        x_test         = splits[4]
        y_test         = splits[5]
        x_val          = splits[6]
        y_val          = splits[7]

        """
        creating seperate lists for each slave to send
        """
        df_xuniques = []
        for i in x_train_unique:
            df_xuniques.append(pd.DataFrame(i))

        df_yuniques = []
        for i in y_train_unique:
            df_yuniques.append(pd.DataFrame(i))

        df_commons = []
        df_commons.append(x_train_common)
        df_commons.append(y_train_common)

        df_tests = []
        df_tests.append(x_test)
        df_tests.append(y_test)

        df_vals = []
        df_vals.append(x_val)
        df_vals.append(y_val)

        """
        Send Unique X Data to slaves. Sending index (1,2,3...) for 
        identifying X columns by the slaves
        """
        tasks_x_unique = []
        for index, data_item in enumerate(df_xuniques, start=1):
            task = asyncio.create_task(self.send(data_item, index))
            tasks_x_unique.append(task)
        done1, pending1 = await asyncio.wait(
            tasks_x_unique, return_when=asyncio.ALL_COMPLETED
        )
        log.p_header(f"X unique sended to all agents.")

        """
        Send Unique Y Data to slaves. Sending index * 10 for 
        identifying ground truth(y) by the slaves
        """
        tasks_y_unique = []
        for index, data_item in enumerate(df_yuniques, start=1):
            task = asyncio.create_task(self.send(data_item, index * 10))
            tasks_y_unique.append(task)
        print(f"Sending {len(tasks_y_unique)} unique datasets")
        done2, pending2 = await asyncio.wait(
            tasks_y_unique, return_when=asyncio.ALL_COMPLETED
        )
        log.p_header(f"Y unique sended to all agents.")
        self.unique_data_sent = True

        """
        Send unique X data to slaves
        Sending 0 for identifiying common dataset by the slaves
        """
        tasks_common = []
        for index, data_item in enumerate(df_commons):
            task = asyncio.create_task(self.send(data_item, 0))
            tasks_common.append(task)
        done3, pending3 = await asyncio.wait(
            tasks_common, return_when=asyncio.ALL_COMPLETED
        )
        self.common_train_data_sent = True

        """
        Send test data to slaves
        """

        tasks = []
        for index, data_item in enumerate(df_tests):
            task = asyncio.create_task(self.send(data_item, -1))
            tasks.append(task)
        done4, pending4 = await asyncio.wait(
            tasks, return_when=asyncio.ALL_COMPLETED
        )
        log.p_header(f"Test data sended to all agents.")

        """
        Send validation data to slaves
        """
        for index, data_item in enumerate(df_vals):
            task = asyncio.create_task(self.send(data_item, -2))
            tasks.append(task)
        done5, pending5 = await asyncio.wait(
            tasks, return_when=asyncio.ALL_COMPLETED
        )
        log.p_header(f"Validation data sended to all agents.")
        return
    
    async def read_from_slaves(self):
        import aiofiles
        """
        Read from slaves
        """
        while True:
            await asyncio.sleep(1)
            file_name = f'models/{self.id}_gathered_metrics.json'
            async with aiofiles.open(file_name, mode="r") as f:
                data = await f.read()
                # if data includes all slaves
                if self.slave_count <= len(data):
                    log.p_ok(f"{log.p_bold(self.id)} All slaves read")
                    break
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
                            # TODO: only get current master bound slave metrics
                            if "id" in decoded_dict:
                                async with aiofiles.open(file_name, mode="a+") as f:
                                    await f.write(json.dumps(decoded_dict) + ",\n")
                                # ack
                                await self.r.xack(self.stream_name, self.group_name, message_id)


                            
            except Exception as e:
                log.p_fail(f"Read slaves fail: {log.p_bold(self.id)} {e}")
                log.p_fail(e.__traceback__.tb_lineno)

        """ 
        log.p_header(f"Reading from slaves")
        tasks = []
        for i in range(self.slave_count):
            task = asyncio.create_task(self.read())
            tasks.append(task)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        log.p_header(f"Read from all slaves.\n{done}\n{pending}\n")
        return """

    async def master_main(self):
        await self.connect_to_redis()
        await self.create_consumer_group()
        log.p_ok(f"{log.p_bold(self.id)} Slave count: {self.slave_count}")

        if not os.path.exists('output'):
            # create output dir
            os.makedirs("output")

            splits = self.split(
                self.read_csv(self.dataset_path),
                number_of_slaves=self.slave_count,
                Y_column_name=self.Y_column_name,
            )

            try:
                await self.send_all(splits)
            except:
                log.p_fail(f"{log.p_bold(self.id)} {e}")
            finally:
                log.p_okblue("Master reading from slaves for metrics")
                await self.read_from_slaves()


            try:
                await self.r.execute_command("XGROUP", "DESTROY", self.stream_name, self.group_name)
            except Exception as e:
                log.p_fail(f"{log.p_bold(self.id)} {e}")

            if self.unique_data_sent:
                log.p_ok(f"{log.p_bold(self.id)} Unique data sent")

            if self.common_train_data_sent:
                log.p_ok(f"{log.p_bold(self.id)} Common data sent")
        else:
            log.p_ok(f"{log.p_bold(self.id)} Output directory exists.\nGoing straigth to training and predicting.")
            print("reading from slaves")
            await self.read_from_slaves()
