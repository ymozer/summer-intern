import pandas as pd
from typing import Protocol
import redis.asyncio as redis
from redis.exceptions import ConnectionError
from log.color import LogColor
log = LogColor()


class Agent():
    def __init__(
        self, id: str, IP: str, port: int | None, stream_name: str, group_name: str
    ) -> None:
        self.id = id
        # Redis instance and attributes
        self.r = None
        self.r_IP = IP
        self.r_port = 6379 if port is None else port
        self.stream_name = stream_name
        self.group_name = group_name
        self.file_opened = False

    def read_csv(self, path):
        df = pd.read_csv(path, sep=";")
        return df

    def read_json(self, path):
        df = pd.read_json(path)
        return df
    
    def decode_list_of_bytes(self, nested_list):
        decoded_list = [
            [item.decode() if isinstance(item, bytes) else item for item in sublist]
            for sublist in nested_list
        ]
        # create a dictionary from even-indexed and odd-indexed pairs for each sublist
        decoded_dict = [dict(zip(even[::2], even[1::2])) for even in decoded_list]
        return decoded_dict

    async def connect_to_redis(self):
        try:
            self.r = redis.Redis(host=self.r_IP, port=self.r_port, db=0)
            # Set the retention time
            max_length = 20000  # Maximum number of messages to retain
            await self.r.xtrim(self.stream_name, maxlen=max_length, approximate=True)
            log.p_ok(
                f"{log.p_bold(self.id)} Connected to Redis at {self.r_IP}:{self.r_port}"
            )
        except ConnectionError as cerr:
            log.p_fail(f"ConnectionError: {cerr}")

    async def write(self, data: dict):
        try:
            await self.r.xadd(self.stream_name, data, "*")
        except Exception as e:
            log.p_fail(f"Redis write Exception: {e}")
            log.p_fail(e.__traceback__.tb_lineno)
            log.p_fail(f"Data: {data}")


    async def create_consumer_group(self):
        try:
            await self.r.xgroup_create(
                self.stream_name, self.group_name, id="0", mkstream=True
            )
            log.p_ok(f"{log.p_bold(self.id)} Created consumer group: {self.group_name}")
        except Exception as e:
            log.p_fail(f"Create customer group Exception: {e}")

    async def info(self):
        info = await self.r.execute_command("XINFO", "GROUPS", self.stream_name)
        log.p_ok(f"{log.p_bold(self.id)} Info: {info}")