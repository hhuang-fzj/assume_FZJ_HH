from assume import World
from assume.scenario.loader_csv import load_scenario_folder

world = World(database_uri=None, export_csv_path="./outputs")
load_scenario_folder(
    world,
    inputs_path="./inputs",\
    scenario="local_retailer_demo",
    study_case="local_retailer_demo",
)
print("场景加载成功，markets:", list(world.markets.keys()))
world.run()
print("仿真跑完")