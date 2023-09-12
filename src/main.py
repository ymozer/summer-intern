import os
import sys
import asyncio
import configparser
import redis.asyncio as redis

from Master import Master
from Slave import Slave
from log.color import LogColor
from utils.file import get_last_character_from_file
from utils.timeit import timeit

from lightgbm import LGBMRegressor
from xgboost  import XGBRegressor

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    Lars,
    LassoLars,
    OrthogonalMatchingPursuit,
    BayesianRidge,
    ARDRegression,
    SGDRegressor,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    TheilSenRegressor,
    HuberRegressor,
    PoissonRegressor,
    TweedieRegressor,
    GammaRegressor,
    )
from sklearn.tree import (
    DecisionTreeRegressor,
    ExtraTreeRegressor
    )
from sklearn.ensemble import (
    ExtraTreesRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    StackingRegressor,
    VotingRegressor,
    IsolationForest,
    HistGradientBoostingRegressor,
    )

from sklearn.neighbors import (
    KNeighborsRegressor,
    RadiusNeighborsRegressor,
    NearestCentroid,
    KernelDensity,
    LocalOutlierFactor,
    )

from sklearn.svm import (
    SVR,
    NuSVR,
    LinearSVR,
    OneClassSVM,
    )

from sklearn.neural_network import (
    MLPRegressor,
    BernoulliRBM,
    )

log=LogColor()
LGBMRegressor()

if __name__ == "__main__":
    # delete models directory and its contents asynchronously
    if os.path.exists("models"):
        for file in os.listdir("models"):
            os.remove(f"models/{file}")
        os.rmdir("models")
        
    ''' delete output directory and its contents
    if os.path.exists("output"):
        for file in os.listdir("output"):
            os.remove(f"output/{file}")
        os.rmdir("output")
    '''
    
    if not os.path.exists("models"):
        os.mkdir("models")

    async def run_different_ratios(ip,
                                   port,
                                   slave_name_list,
                                   slave_model_list,
                                   dataset_path,
                                   y_column_name,
                                   test_ratio=0.2,
                                   validation_ratio=0.1
                                   ):
        '''
        Run master and slave instances for each common-unique ratio in (10,100,10)
        ''' 
        for i in range(10,100,10):
            slave_instances = []
            counter = 0
            common_ratio = i/100
            unique_ratio = round(1-common_ratio,1)
            # fix floating point precision

            master_id=f"m_{common_ratio}_{unique_ratio}"
            master = Master(
                master_id,
                ip,
                port,
                "stream_1",
                "group_1",
                dataset_path=dataset_path,
                common_ratio=common_ratio,
                test_ratio=test_ratio,
                validation_ratio=validation_ratio,
                y_column_name=y_column_name,
                slave_count=len(slave_name_list)
            )

            for slave_ins in slave_name_list:
                ins = Slave(
                    f"{slave_ins}_{master_id}",
                    ip,
                    port,
                    "stream_1",
                    f"group_{counter+2}",
                    slave_model_list[counter],
                    {},
                    3
                )
                slave_instances.append(ins.slave_main())
                counter += 1
                
            open(f"models/{master_id}_gathered_metrics.json", "w").close()
            try:
                await asyncio.gather(master.master_main(), *slave_instances)
            except Exception as e:
                log.p_fail(f"Exception: {e}")
                log.p_fail(e.__traceback__.tb_lineno)
    
    async def run_once(dataset_path,
                       common_ratio,
                       test_ratio,
                       validation_ratio,
                       y_column_name,
                       ip,
                       port,
                       stream_name,
                       master_group_name,
                       slave_name_list:list,
                       slave_model_list:list,
                       delay=0 
                       ):
        '''
        Run master and slave instances for a single common-unique ratio
        '''
        slave_instances = []
        unique_ratio = round(1-common_ratio,1)
        # fix floating point precision
        master_id=f"m_{common_ratio}_{unique_ratio}"
        open(f"models/{master_id}_gathered_metrics.json", "w").close()

        master = Master(
            master_id,
            ip,
            port,
            stream_name,
            master_group_name,
            dataset_path=dataset_path,
            delay=0,
            common_ratio=common_ratio,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
            y_column_name=y_column_name,
            slave_count=len(slave_name_list)
        )

        counter = 0
        for slave_ins in slave_name_list:
            ins = Slave(
                slave_ins,
                ip,
                port,
                stream_name,
                f"group_{counter+2}",
                slave_model_list[counter],
                {},
                delay
            )

            slave_instances.append(ins.slave_main())
            counter += 1
            
        try:
            await asyncio.gather(master.master_main(), *slave_instances)
        except Exception as e:
            log.p_fail(f"Run Once Exception: {e}")
            log.p_fail(e.__traceback__.tb_lineno)


    async def main():        
        config = configparser.ConfigParser()
        config.read("config.ini", encoding="utf-8")
        sections = config.sections()
        slave_name_list = []
        slave_model_list = []
        for section in sections:
            if section.startswith("slave"):
                slave_name_list.append(config.get(section, "name"))
                slave_model_list.append(eval(eval(config.get(section, "model"))))
        
        if config.getboolean("general", "run_different_ratios"):
            try:
                await run_different_ratios(
                    str(config.get("center", "ip")).strip('"'),
                    config.getint("center", "port"),
                    slave_name_list,
                    slave_model_list,
                    config.get     ("center", "dataset_path"),
                    config.get     ("center", "y_column_name"),
                    config.getfloat("center", "test_ratio"),
                    config.getfloat("center", "val_ratio")
                )
            except Exception as e:
                log.p_fail(f"Ratios Exception: {e}")
                log.p_fail(e.__traceback__.tb_lineno)
                sys.exit(1)
        else:
            try:
                await run_once(
                    config.get     ("center", "dataset_path"),
                    config.getfloat("center", "common_ratio"),
                    config.getfloat("center", "test_ratio"),
                    config.getfloat("center", "val_ratio"),
                    config.get     ("center", "y_column_name"),
                    str(config.get("center", "ip")).strip('"'),
                    config.getint  ("center", "port"),
                    config.get     ("center", "stream"),
                    config.get     ("center", "group"),
                    slave_name_list,
                    slave_model_list,
                    config.getint  ("center", "delay_read")
                )
            except Exception as e:
                log.p_fail(f"Run Once Exception: {e}")
                log.p_fail(e.__traceback__.tb_lineno)
                sys.exit(1)

    # if system is macos, use uvloop
    if sys.platform == "darwin":
        import uvloop
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            try:
                runner.run(main())
            finally:
                log.p_warn("QUITTING")
                sys.exit(1)

    elif sys.platform == "win32":
        with asyncio.Runner() as runner:
            try:
                runner.run(main())
            finally:
                log.p_warn("QUITTING")
                sys.exit(1)
    else :
        log.p_fail("Unsupported OS")

