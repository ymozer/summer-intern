from Agent import Agent
from asyncio import sleep
import random
import time 
import datetime

class Master(Agent):
    def __init__(self, id: str, IP: str, port:int|None, stream_name:str, group_name:str) -> None:
         super().__init__(id, IP, port, stream_name, group_name)


    async def main(self):
        await self.connect_to_redis()
        await self.create_consumer_group()
        

        while True:
            # RNG
            data = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'id': random.randint(0, 100),
                'data': random.randint(0, 100)
            }
            await self.write(data)
            await sleep(1)