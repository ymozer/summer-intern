from Agent import Agent
from log.color import LogColor

import time 
import json
import random
import datetime
import asyncio
import pandas as pd
import numpy as np
from asyncio import sleep
from sklearn.model_selection import train_test_split

log=LogColor()


class Master(Agent):
    def __init__(self, id: str, IP: str, port:int|None, stream_name:str, group_name:str, dataset_path:str) -> None:
         super().__init__(id, IP, port, stream_name, group_name)
         self.dataset_path = dataset_path
         self.slave_count=0

    def decode_list_of_bytes(self,nested_list):
        decoded_list = [
            [item.decode() if isinstance(item, bytes) else item for item in sublist]
            for sublist in nested_list
        ]
        #create a dictionary from even-indexed and odd-indexed pairs for each sublist
        decoded_dict = [
            dict(zip(even[::2], even[1::2])) for even in decoded_list
        ]
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
        info = await self.r.execute_command('XINFO', 'GROUPS', self.stream_name)
        
        dictt = self.decode_list_of_bytes(info)
        json_data = json.dumps(dictt, indent=4)
        # pretty print
        log.p_ok(f"data: {json_data}")
        
        # check if consumers are bigger than 0
        counter=0
        for i in dictt:
            if i['consumers'] > 0:
                counter+=1
        
        # bacause master has its own group
        self.slave_count=counter

    def read_csv(self, path):
        df = pd.read_csv(path,sep=';')
        return df
    
    def read_json(self, path):
        df = pd.read_json(path)
        return df
    
    def split(self, df, 
              number_of_slaves:int=2, 
              Y_column_name:str='LV ActivePower (kW)', 
              test_ratio:float=0.2, 
              common_ratio:float=0.5, 
              random_state:int=42, 
              shuffle:bool=True):
        """
        Split the data into training and testing sets
        """
        X = df.drop(columns=Y_column_name, axis=1)
        y = df[Y_column_name]
        x_train, x_text, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state, shuffle=shuffle)
        
        # split training again for unique and common
        x_train_unique, x_train_common, y_train_unique, y_train_common = train_test_split(x_train, y_train, test_size=common_ratio, random_state=random_state, shuffle=shuffle) 
        
        # split again for each number of slaves
        x_train_unique_split = np.array_split(x_train_unique, number_of_slaves)
        y_train_unique_split = np.array_split(y_train_unique, number_of_slaves)
        log.p_ok(f"{log.p_bold(self.id)} Unique data split into {len(x_train_unique_split)} parts")
        for i in range(number_of_slaves):
            ''' Add identifier column for slaves '''
            num_rows = x_train_unique_split[i].shape[0]
            new_column = np.full((num_rows, 1), i+1)
            x_train_unique_split[i] = np.hstack((x_train_unique_split[i], new_column))

        return x_train_unique_split, y_train_unique_split,\
            x_train_common, y_train_common,\
            x_text, y_test
    
    async def send(self,dataTsend,agentid:int):
        for index, row in dataTsend.iterrows():
            await self.write(row.to_dict())

    async def send_all(self,splits):
        x_train_unique=splits[0]
        df_uniques=[]
        for i in x_train_unique:
            df_uniques.append(pd.DataFrame(i))
        log.p_warn(f"Number of unique datasets:\t{len(df_uniques)}")
        await asyncio.gather(
            self.send(df_uniques[0],1),
            self.send(df_uniques[1],2),
            self.send(df_uniques[2],3)
        )


    async def main(self):
        await self.connect_to_redis()
        await self.create_consumer_group()
        await self.slave_counter()
        log.p_ok(f"{log.p_bold(self.id)} Slave count: {self.slave_count}")
        splits=self.split(
            self.read_csv(self.dataset_path),
            number_of_slaves=self.slave_count,
            Y_column_name='LV ActivePower (kW)',
            test_ratio=0.2,
            common_ratio=0.5,
            random_state=42,
            shuffle=True
            )
        while True:
            await self.send_all(splits)
            await sleep(4)