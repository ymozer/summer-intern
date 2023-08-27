import os
import sys
import asyncio
import redis.asyncio as redis

# Redis exceptions
from redis.exceptions import ConnectionError, TimeoutError
import uvloop
import argparse

from Agent import Agent
from Master import Master
from Slave import Slave
from log.color import LogColor
from utils.file import get_last_character_from_file
from utils.timeit import timeit

from lightgbm import LGBMRegressor
from xgboost  import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)

log=LogColor()

if __name__ == "__main__":
    slave_list = ["agent1", "agent2", "agent3", "agent4"]
    model_names=[ LGBMRegressor(), XGBRegressor(), RandomForestRegressor(), DecisionTreeRegressor(), AdaBoostRegressor(), ExtraTreesRegressor()]
    parser = argparse.ArgumentParser(description="Agent")
    parser.add_argument("--id", type=str, default="agent_1", help="Agent ID")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Redis IP")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--slave",
        type=str,
        nargs="+",
        default=slave_list,
        help="Slave names separated by space",
    )

    if not os.path.exists("models"):
        os.mkdir("models")


    async def main():
        args = parser.parse_args()


        """
        master = Master(
            "master",
            args.ip,
            args.port,
            "stream_1",
            "group_1",
            dataset_path="T1_short.csv", #dataset_path="T1.csv",
            delay=0,
            common_ratio=0.2,
            test_ratio=0.2,
            validation_ratio=0.2,
        )

        await asyncio.gather(master.master_main(), *slave_instances)
        """
        for i in range(10,100,10):
            slave_instances = []
            counter = 0
            common_ratio = i/100
            unique_ratio = round(1-common_ratio,1)
            # fix floating point precision

            master_id=f"m_{common_ratio}_{unique_ratio}"
            master = Master(
                master_id,
                args.ip,
                args.port,
                "stream_1",
                "group_1",
                dataset_path="T1.csv",#dataset_path="T1_short.csv",
                delay=0,
                common_ratio=common_ratio,
                test_ratio=0.2,
                validation_ratio=0.1,
            )
            for slave_ins in args.slave:
                ins = Slave(f"{slave_ins}_{master_id}", args.ip, args.port, "stream_1", f"group_{counter+2}", model_names[counter],{})
                slave_instances.append(ins.slave_main())
                counter += 1
                
            open(f"models/{master_id}_gathered_metrics.json", "w").close()
            try:
                await asyncio.gather(master.master_main(), *slave_instances)
            except Exception as e:
                log.p_fail(f"Exception: {e}")
                log.p_fail(e.__traceback__.tb_lineno)



    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        try:
            runner.run(main())
        finally:
            log.p_warn("QUITTING")
            sys.exit(1)
