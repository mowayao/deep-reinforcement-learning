

class defaults:
    steps_per_epoch = 250000
    epoch_num = 300
    steps_per_test = 125000
    replay_start_size = 50000
    memory_size = 1000000
    phi_len = 4
    action_repeat = 4
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 1000000
    no_op_max = 30
    resized_width = 84
    resized_height = 84
    base_rom_path = "roms/"
    exps_prefix = "exps/"
    rom = 'breakout.bin'
    frame_skip = 1
    display_screen = False
    batch_size = 32
    learning_rate = 0.00025
    rmsp_rho = 0.95
    rmsp_epsilon = 0.01
    discount = 0.99
    update_frequency = 4
    target_nn_update_frequency = 10000
    repeat_action_probability = 0.0
    death_end_episode = False
    buffer_size = 200
    holdout_data_size = 4000


	
