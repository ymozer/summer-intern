from Agent import Agent


class Slave(Agent):
    def __init__(self, id: str, IP: str, port:int|None, stream_name:str, group_name:str) -> None:
         super().__init__(id, IP, port, stream_name, group_name)


    async def main(self):
        await self.connect_to_redis()
        await self.create_consumer_group()
        await self.read()


    
