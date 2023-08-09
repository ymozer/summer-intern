import os
import sys
import time
import json
import asyncio
import aiofiles
import argparse
import redis.asyncio as redis
# Redis exceptions
from redis.exceptions import ConnectionError,TimeoutError

from sklearn.model_selection import train_test_split
from io import BytesIO
import pandas as pd
import numpy as np

from log.color import LogColor 
from utils.file import file_control
from utils.timeit import timeit


log=LogColor()

class Agent:
    def __init__(self, id: str, IP: str, port:int|None, stream_name:str, group_name:str) -> None:
        self.id = id
        # Redis instance and attributes
        self.r = None
        self.r_IP=IP
        self.r_port= 6379 if port is None else port
        self.stream_name = stream_name
        self.group_name = group_name
        self.file_opened = False
    
    def read_csv(self, path):
        df = pd.read_csv(path,sep=';')
        return df
    
    def read_json(self, path):
        df = pd.read_json(path)
        return df

    async def connect_to_redis(self):
        try:
            self.r = redis.Redis(
                host=self.r_IP,
                port=self.r_port, 
                db=0
            )
            # Set the retention time
            max_length = 20000  # Maximum number of messages to retain
            self.r.xtrim(self.stream_name, maxlen=max_length, approximate=True)
            log.p_ok(f"{log.p_bold(self.id)} Connected to Redis at {self.r_IP}:{self.r_port}")
            
        except ConnectionError as cerr:
            log.p_fail(f"ConnectionError: {cerr}")

    async def write(self, data: dict):
        try:
            await self.r.xadd(self.stream_name, data, "*")
        except Exception as e:
            log.p_fail(f"Write Exception: {e}")

    async def create_consumer_group(self):
        try:
            await self.r.xgroup_create(self.stream_name, self.group_name, id='0', mkstream=True)
            log.p_ok(f"{log.p_bold(self.id)} Created consumer group: {self.group_name}")
        except Exception as e:
            log.p_fail(f"Create customer group Exception: {e}")
    
    async def info(self):
        info = await self.r.execute_command('XINFO', 'GROUPS', self.stream_name)
        log.p_ok(f"{log.p_bold(self.id)} Info: {info}")


    async def read(self):
        while True:
            file_name = f"output/{self.id}_xunique_train.json"
            try:
                response =  await self.r.xreadgroup(self.group_name, self.stream_name, {self.stream_name: '>'}, None)
                if response:
                    for stream_name, stream_data in response:
                        for message_id, message_data in stream_data:
                            decoded_dict = {key.decode(): value.decode() for key, value in message_data.items()}
                            last_col=decoded_dict['999']
                            if last_col == 'END':
                                break
                            flag_val=int(float(last_col))
                            if flag_val == int(self.id[-1]) or flag_val == 0 or flag_val == int(self.id[-1])*10 or flag_val == int(79):
                                if flag_val == 0:
                                    file_name = f"output/{self.id}_xcommon.json"
                                elif flag_val == int(79):
                                    file_name = f"output/{self.id}_ycommon.json"
                                elif flag_val == int(self.id[-1])*10:
                                    file_name = f"output/{self.id}_yunique_train.json"
                                json_string = json.dumps(decoded_dict)
                                # write to file using aiofiles
                                async with aiofiles.open(file_name, mode='a') as f:
                                    # if file is empty, write the first line as [
                                    if os.stat(file_name).st_size == 0:
                                        await f.write("[\n")
                                    file_control(file_name,self.file_opened)
                                    self.file_opened = True
                                    await f.write(json_string+",\n")
                                await self.r.xack(stream_name, self.group_name, message_id)
            except Exception as e:
                log.p_fail(f"{log.p_bold(self.id)} Redis Read Exception: {e.with_traceback(e.__traceback__)}{e.__cause__}")
                await asyncio.sleep(1)



    
            