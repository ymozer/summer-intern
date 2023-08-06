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

    async def connect_to_redis(self):
        try:
            self.r = redis.Redis(
                host=self.r_IP,
                port=self.r_port, 
                db=0
            )
            # Set the retention time
            #max_length = 100  # Maximum number of messages to retain
            #self.r.xtrim(self.stream_name, maxlen=max_length, approximate=True)

            log.p_ok(f"{log.p_bold(self.id)} Connected to Redis at {self.r_IP}:{self.r_port}")
        except ConnectionError as cerr:
            log.p_fail(f"ConnectionError: {cerr}")

    async def write(self, data: dict):
        try:
            await self.r.xadd(self.stream_name, data, "*")
            #log.p_ok(f"{log.p_bold(self.id)} Data written to stream: {data}")
        except Exception as e:
            log.p_fail(f"Exception: {e}")

    async def create_consumer_group(self):
        try:
            await self.r.xgroup_create(self.stream_name, self.group_name, id='0', mkstream=True)
            log.p_ok(f"{log.p_bold(self.id)} Created consumer group: {self.group_name}")
        except Exception as e:
            log.p_fail(f"Exception: {e}")

    async def read(self):
        """
        https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.cluster.RedisClusterCommands.xgroup_create
        args:
            - name: name of the stream to read from
            - groupname : name of the consumer group to read from
            - id : ID of the last item in the stream to consider already delivered.
            
        """
        while True:
            response = None
            try:
                response =  await self.r.xreadgroup(self.group_name, self.stream_name, {self.stream_name: '>'}, None)
                await asyncio.sleep(1)
                if response:
                    for stream_name, stream_data in response:
                        for message_id, message_data in stream_data:
                            decoded_dict = {key.decode(): value.decode() for key, value in message_data.items()}
                            json_string = json.dumps(decoded_dict)
                            log.p_ok(f"{log.p_bold(self.id)} Received message: {json_string}")
                            # write to file using aiofiles
                            async with aiofiles.open(f"output/{self.id}.json", mode='a') as f:
                                # if file is empty, write the first line as [
                                if os.stat(f"output/{self.id}.json").st_size == 0:
                                    await f.write("[\n")
                                file_control(f"output/{self.id}.json",self.file_opened)
                                self.file_opened = True
                                await f.write(json_string+",\n")
                            #await self.r.xack(stream_name, self.group_name, message_id)
            except Exception as e:
                log.p_fail(f"{log.p_bold(self.id)} Exception: {e.with_traceback(e.__traceback__)}")
                await asyncio.sleep(1)



    
            