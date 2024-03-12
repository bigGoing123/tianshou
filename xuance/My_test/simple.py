import xuance
runner = xuance.get_runner(method='maddpg',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=True)
runner.run()