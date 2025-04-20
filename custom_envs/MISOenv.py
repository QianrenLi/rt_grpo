import numpy as np
from .channelKnowledgeMap import MapGenerator
from numpy import linalg as LA

import os
import importlib.util
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MISOEnvWrapper(gym.Env):
    def __init__(self, config_folder_path = None, **kwargs):
        super().__init__()
        if config_folder_path is None:
            raise ValueError("config_folder_path must be provided")
        
        config_files = [os.path.join(config_folder_path, f) for f in os.listdir(config_folder_path) if f.endswith(".json")]
        exp_config_path = os.path.join(config_folder_path, "exp_config.py")
        if not os.path.exists(exp_config_path):
            raise FileNotFoundError(f"exp_config.py not found in {config_folder_path}")

        # Import the Python module dynamically
        spec = importlib.util.spec_from_file_location("exp_config", exp_config_path)
        exp_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp_config)
        
        is_enjoy = kwargs.get("render_mode", None)
        self.env = MISOEnv(config_files, exp_config, is_enjoy = is_enjoy)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(self.env.action_dim,), dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=-1000, high=1000, shape=(self.env.observation_dim,), dtype=np.float64
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done = self.env.step(action)
        return obs, reward, done, False, {}

    def render(self, mode="human"):
        print(f"TX Position: {self.env.tx_position}")
        print(f"Beamforming: {self.env.beamforming}")
        
    def tx_position(self):
        return self.env.tx_position

    def close(self):
        self.env.close()


def complex_array_to_real_imag(array):
    return np.concatenate([array.real, array.imag])

def real_imag_array_to_complex(array):
    return array[:len(array)//2] + 1j * array[len(array)//2:]

class MISOEnv:
    def __init__(self, config_files, exp_config, is_enjoy = None):
        self.is_enjoy = is_enjoy
        
        self.rx_channel_handles = [ MapGenerator(config_file) for config_file in config_files ]
        self.channel_num = len(self.rx_channel_handles)
        
        self.ref_rx_x = self.rx_channel_handles[0].tx_x
        self.ref_rx_y  = self.rx_channel_handles[0].tx_y
        self.tx_y = exp_config.y_pos
        self.antenna_num = exp_config.antenna_num
        
        self.theta_l = self.x_theta_conversion(exp_config.x_pos_l)
        self.theta_r = self.x_theta_conversion(exp_config.x_pos_r)
        
        theta_0 = self.x_theta_conversion( (exp_config.x_pos_l + exp_config.x_pos_r) / 2 )
        self.x_thetas = np.array([ theta_0 + i * exp_config.delta_theta for i in range(exp_config.antenna_num) ])

        self.tx_position = np.array( [ np.array([ self.theta_x_conversion(self.x_thetas[i]), exp_config.y_pos], dtype=float) for i in range(exp_config.antenna_num) ] )        
        
        self.beamforming = np.zeros((len(self.rx_channel_handles), exp_config.antenna_num))
        self.beamforming_state = np.zeros(self.channel_num * exp_config.antenna_num * 2)
        self.power_constraint = getattr(exp_config, 'power_constraint', 1.0)
        
        self.delta_theta_max = np.pi / 90
        
        self.t = 0
        self.max_t = 50
        
        self.his_reward = []
        self.maximum_reward_num = 10
        self.observation_dim = self.channel_num * exp_config.antenna_num * 4 + 1
        
        self.exp_config = exp_config
        
        self.reset()
        
    def theta_x_conversion(self, theta_x ):
        return self.ref_rx_x - (abs(self.ref_rx_y - self.tx_y)) / np.tan(theta_x)

    def x_theta_conversion(self, x ):
        theta = np.arctan( (self.ref_rx_y - self.tx_y) / (self.ref_rx_x - x)  )
        if theta < 0:
            theta += np.pi
        return theta
    
    def add_theta_x(self, x, delta_theta):
        return self.theta_x_conversion(self.x_theta_conversion(x) + delta_theta)
    
    @property
    def action_dim(self):
        return self.channel_num * self.exp_config.antenna_num * 2 + self.exp_config.antenna_num
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def reset(self):
        theta_0 = self.x_theta_conversion( (self.exp_config.x_pos_l + self.exp_config.x_pos_r) / 2 )
        self.x_thetas = np.array([ theta_0 + i * self.exp_config.delta_theta for i in range(self.exp_config.antenna_num) ])
        self.tx_position = np.array( [ np.array([ self.theta_x_conversion(self.x_thetas[i]), self.exp_config.y_pos], dtype=float) for i in range(self.exp_config.antenna_num) ] )
        self.beamforming = np.zeros((len(self.rx_channel_handles), self.exp_config.antenna_num), dtype=complex)
        state, _, _ = self.step(np.zeros(self.action_dim))
        self.t = 0
        self.observation_dim = len(state)
        return state
    
    def reward_func(self):
        """_summary_

        Returns:
            h_list([ND.array]) : numUsr * numTx
        """
        reward = 0
        h_list = []
        signal_powers = []

        for i, rx_channel in enumerate(self.rx_channel_handles):
            signal_power = []
            h_vec = []
            for j in range(self.exp_config.antenna_num):
                h = rx_channel.generate_h(self.tx_position[j, 0], self.tx_position[j, 1], self.exp_config.mode)
                h_vec.append(h)
            
            h_vec = np.array(h_vec)
            h_list.append(h_vec) # Assume the same location for all antennas
            for channel_idx in range(len(self.rx_channel_handles)):
                signal_power.append(np.abs( self.beamforming[channel_idx] @ h_vec.conj().T ) ** 2)
                
            signal_powers.append(signal_power)
        rate = 0
        for i in range(len(self.rx_channel_handles)):
            signal_power = signal_powers[i]
            rate += np.log2(1 + (signal_power[i] / (self.exp_config.sigma2 + np.sum(signal_power) - signal_power[i])))
        reward = rate

        return h_list, reward
    
    def revise_theta(self):        
        for i in range(len(self.x_thetas)):
            lower_bound = self.theta_l + i * self.exp_config.delta_theta
            upper_bound = self.theta_r - (len(self.x_thetas) - i + 1) * self.exp_config.delta_theta
            if self.x_thetas[i] < lower_bound:
                self.x_thetas[i] = lower_bound
            if self.x_thetas[i] > upper_bound:
                self.x_thetas[i] = upper_bound

        for i in range(len(self.tx_position) - 1):
            delta_theta = self.x_thetas[i + 1] - self.x_thetas[i]
            if delta_theta < self.exp_config.delta_theta:
                self.x_thetas[i + 1] = self.x_thetas[i] + self.exp_config.delta_theta

        self.tx_position = np.array( [ np.array([ self.theta_x_conversion(self.x_thetas[i]), self.exp_config.y_pos], dtype=float) for i in range(self.exp_config.antenna_num) ] )
        
    def revise_beamforming(self):
        self.beamforming = real_imag_array_to_complex(self.beamforming_state).reshape((len(self.rx_channel_handles), self.exp_config.antenna_num))
        consumed_power = np.sum( self.beamforming @ self.beamforming.conj().T )
        if consumed_power > self.power_constraint:
            self.beamforming = self.beamforming * np.sqrt(self.power_constraint / consumed_power) 
        return consumed_power
    
    def step(self, action):
        # Update the tx_position and beamforming
        # self.x_thetas += action[-env_config.antenna_num:] * 0.1
        # self.beamforming_state += action[:-env_config.antenna_num] * 0.1
        
        self.x_thetas = (action[-self.exp_config.antenna_num:] - min(action[-self.exp_config.antenna_num:])) * 0.01 + self.theta_l
        self.beamforming_state = action[:-self.exp_config.antenna_num]
        
        self.revise_theta()
        self.revise_beamforming()

        h_list, reward = self.reward_func()
        self.t += 1
        self.beamforming_state = complex_array_to_real_imag(self.beamforming.flatten())

        # if self.is_enjoy is not None:
        #     print(f"TX Position: {self.tx_position}")
        #     print(f"Beamforming: {self.beamforming}")

        if self.t == self.max_t:
            return complex_array_to_real_imag(np.array(h_list).flatten()), reward, True
        return complex_array_to_real_imag(np.array(h_list).flatten()), reward, False
    
    def close(self):
        pass


if __name__ == "__main__":
    import os
    config_folders = os.listdir("custom_envs/configs/exp-change_antenna_num")
    for folder in config_folders:
        config_folder_path = os.path.join("custom_envs/configs/exp-change_antenna_num", folder)
        break
    print(config_folder_path)
    env = MISOEnvWrapper(config_folder_path).env
    exp_config = env.exp_config
    print()
    actions = np.random.randn(env.action_dim) 
    print(actions[-exp_config.antenna_num:])
    h_list,_,_ = env.step(actions)
    print(env.tx_position)
    print(h_list)
    
    # from custom_envs.MMSE import WMMSE
    # solver = WMMSE()
    # h_list = real_imag_array_to_complex(h_list).reshape((len(env.rx_channel_handles), exp_config.antenna_num))
    # H = []
    # for idx in range(len(h_list)):
    #     H.append(h_list[idx].reshape((1, -1)))
        
    # V, rate = solver.solve(H)
    # precoder = []
    # for idx in range(len(V)):
    #     precoder.append(V[idx].flatten())
    # precoder = np.array(precoder)
    
    # print(precoder, rate)
    # print(H)
    # ##
    # consumed_power = np.sum( precoder @ precoder.conj().T )
    # env.beamforming  = precoder * np.sqrt(1 / consumed_power)
    # print(env.beamforming)
    # print(consumed_power)
    # print(env.reward_func())