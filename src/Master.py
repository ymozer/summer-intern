from Agent import Agent
from log.color import LogColor
from utils.timeit import timeit, async_timeit
from varname import varname, nameof

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
    ) -> None:
        super().__init__(id, IP, port, stream_name, group_name)
        self.dataset_path = dataset_path
        self.slave_count = 0
        self.unique_data_sent = False
        self.delay = delay
        self.common_train_data_sent = False

    def decode_list_of_bytes(self, nested_list):
        decoded_list = [
            [item.decode() if isinstance(item, bytes) else item for item in sublist]
            for sublist in nested_list
        ]
        # create a dictionary from even-indexed and odd-indexed pairs for each sublist
        decoded_dict = [dict(zip(even[::2], even[1::2])) for even in decoded_list]
        return decoded_dict

    async def slave_counter(self):
        """
        [{'name': 'group_1',
        'consumers': 0,
        'pending': 0,
        'last-delivered-id': '0-0',
        'entries-read': None,
        'lag': 0}]
        """
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

        with open("temp.txt", "a+") as f:
            file_data = f.read()
            print("file data: ", file_data)
            if file_data == "true":
                self.slave_count = counter
                return
            else:
                f.write(f"true\n")
                self.slave_count = len(dictt) - 1

        log.p_ok(f"slave count: {self.slave_count}")
        # create a file for keeping log on is this file executed before

    def split(
        self,
        df,
        number_of_slaves: int = 2,
        Y_column_name: str = "LV ActivePower (kW)",
        validation_ratio: float = 0.2,
        test_ratio: float = 0.2,
        common_ratio: float = 0.2,
        random_state: int = 42,
        shuffle: bool = True,
    ):
        """
        Split the data into training and testing sets
        """
        X = df.drop(columns=Y_column_name, axis=1)
        y = df[Y_column_name]
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio,
            random_state=random_state, shuffle=shuffle
        )


        # split training data into training and validation
        # 0.25 x 0.8 = 0.2
        (x_train, x_val, y_train, y_val) = train_test_split(
            x_train, y_train, test_size=validation_ratio, 
            random_state=random_state, shuffle=shuffle
        ) 

        # split training again for unique and common
        (x_train_unique, x_train_common,
          y_train_unique,y_train_common) = train_test_split(
            x_train, y_train, test_size=common_ratio,
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

            x_train_unique_split[i].rename(columns={4: "999"}, inplace=True)
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
        y_train_common["999"] = 79  # LV ActivePower (kW)

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
        if agentid == -2:
            log.p_warn(f"Validation dataset shape:\t{dataTsend.shape}")
        if agentid == -1:
            log.p_warn(f"test dataset shape:\t\t{dataTsend.shape}")
        elif agentid == 0:
            log.p_warn(f"Common dataset shape:\t\t{dataTsend.shape}")
        elif agentid % 10 == 0:
            log.p_warn(f"y training dataset shape:\t{dataTsend.shape}")
        else:
            log.p_warn(f"Agent{agentid} dataset shape:\t{dataTsend.shape}")

        try:
            for index, row in dataTsend.iterrows():
                await self.write(row.to_dict())
        except Exception as e:
            log.p_fail(f"Redis send exception: {log.p_bold(self.id)} {e}")
        return

    async def send_all(self, splits):
        """
        I want to make each sending process blocking. For example:
        First send x_train_unique
        Second send y_train_unique and so on

        Need to figure out slowing sending process down.
        """
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
        print(f"Sending {len(tasks_x_unique)} unique datasets")
        done1, pending1 = await asyncio.wait(
            tasks_x_unique, return_when=asyncio.ALL_COMPLETED
        )
        log.p_header(f"X unique sended to all agents.\n{done1}\n{pending1}\n")

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
        log.p_header(f"Y unique sended to all agents.\n{done2}\n{pending2}\n")
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
        log.p_header(f"Test data sended to all agents.\n{done4}\n{pending4}\n")

        """
        Send validation data to slaves
        """
        for index, data_item in enumerate(df_vals):
            task = asyncio.create_task(self.send(data_item, -2))
            tasks.append(task)
        done5, pending5 = await asyncio.wait(
            tasks, return_when=asyncio.ALL_COMPLETED
        )
        log.p_header(f"Validation data sended to all agents.\n{done5}\n{pending5}\n")
        return

    @async_timeit
    async def master_main(self):
        await self.connect_to_redis()
        await self.create_consumer_group()
        await self.slave_counter()
        log.p_ok(f"{log.p_bold(self.id)} Slave count: {self.slave_count}")

        if not os.path.exists('output'):
            # create output dir
            if not os.path.exists("output"):
                os.makedirs("output")

            splits = self.split(
                self.read_csv(self.dataset_path),
                number_of_slaves=self.slave_count,
                Y_column_name="LV ActivePower (kW)",
                test_ratio=0.2,
                common_ratio=0.5,
                random_state=42,
                shuffle=True,
            )

            try:
                task = asyncio.create_task(self.send_all(splits))
                await task
                if task.done():
                    task.cancel()
            except Exception as e:
                log.p_fail(f"{log.p_bold(self.id)} {e}")

            if self.unique_data_sent:
                log.p_ok(f"{log.p_bold(self.id)} Unique data sent")

            if self.common_train_data_sent:
                log.p_ok(f"{log.p_bold(self.id)} Common data sent")
        else:
            pass
