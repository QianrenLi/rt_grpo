import numpy as np
import json
import time
import os
from .poliLine import Polyline

class MapGenerator:
    def __init__(self, config_file, epsilon=8, mode=1):
        # Load the configuration
        self.config = self.load_config(config_file)
        self.corner_x = np.array(self.config["corner_x"])
        self.corner_y = np.array(self.config["corner_y"])
        self.tx_x, self.tx_y = self.config["tx"]
        self.f = self.config["f"]
        self.epsilon = epsilon
        self.mode = mode
        self.c = 3e8  # Speed of light (m/s)
        self.lambda_ = self.c / self.f  # Wavelength (meters)
        self.sigma2 = 1e-9  # AWGN Power
        self.G_t, self.G_r = 1, 1  # Transmitter and Receiver gains
        self.poly = Polyline(self.corner_x, self.corner_y)  # Create Polyline object

        # Calculate mirrored points once and store
        self.mirror_x, self.mirror_y = self.poly.compute_mirrored_points(self.tx_x, self.tx_y)

    def load_config(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def cal_gamma_layout(self, alpha, epsilon, mode):
        if mode == 1:  # TE Mode
            return (np.sin(alpha) - np.sqrt(epsilon - np.cos(alpha) ** 2)) / (
                np.sin(alpha) + np.sqrt(epsilon - np.cos(alpha) ** 2)
            )
        else:  # TM Mode
            return (-epsilon * np.sin(alpha) + np.sqrt(epsilon - np.cos(alpha) ** 2)) / (
                epsilon * np.sin(alpha) + np.sqrt(epsilon - np.cos(alpha) ** 2)
            )

    def generate_h(self, rx_x, rx_y, mode):
        """
        Calculate the complex path loss component h for a given receiver point.
        """
        # Check for ray existence and calculate alpha
        ray_existence, alpha = self.poly.check_ray_existence(
            self.tx_x, self.tx_y, self.mirror_x, self.mirror_y, rx_x, rx_y
        )

        multi_path_effect = 0

        # Line-of-sight (LOS) component
        los_existence = self.poly.check_los_ray_existence(self.tx_x, self.tx_y, rx_x, rx_y)
        if los_existence.any():
            d_los = np.linalg.norm([rx_x - self.tx_x, rx_y - self.tx_y])
            multi_path_effect += np.exp(-1j * 2 * np.pi * d_los / self.lambda_) / d_los

        # Non-line-of-sight (NLOS) component (multiple reflections)
        if ray_existence.any():
            alpha_valid = alpha[ray_existence]
            mirror_x_valid = self.mirror_x[ray_existence]
            mirror_y_valid = self.mirror_y[ray_existence]
            gammas = self.cal_gamma_layout(alpha_valid, self.epsilon, mode)
            d_nlos = np.linalg.norm(
                np.column_stack([rx_x - mirror_x_valid, rx_y - mirror_y_valid]), axis=1
            )
            multi_path_effect += np.sum(
                gammas * np.exp(-1j * 2 * np.pi * d_nlos / self.lambda_) / d_nlos
            )

        # Path loss model: h based on the transmitter/receiver gains and multipath effects
        h = (np.sqrt(self.G_t * self.G_r) * self.lambda_ / (4 * np.pi)) * multi_path_effect
        return h

    def generate_map(self):
        print(f"=====Generate Map {self.config['name']}=====")
        start_time = time.time()

        # Iterate over points and compute SNR
        points = self.poly.iterate_points(0.05)
        snr_array = np.zeros((3, len(points)))
        print_interval = 100
        count = 0

        for point in points:
            count += 1
            rx_x, rx_y = point

            # Calculate 'h' for this point
            h = self.generate_h(rx_x, rx_y, self.mode)

            # Calculate the pathloss (absolute value of h)
            pathloss = np.abs(h)

            # Store the results in the SNR array
            snr_array[0, count - 1] = rx_x
            snr_array[1, count - 1] = rx_y
            snr_array[2, count - 1] = pathloss

            if count % print_interval == 0:
                print(f"Progress: {100 * count / len(points):.2f}%")

        print("End")
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

        # Save results
        result_dir = "data"
        os.makedirs(result_dir, exist_ok=True)
        np.save(f"{result_dir}/{self.config['name']}.npy", snr_array)

# Example of usage
if __name__ == "__main__":
    # config_file = "custom_envs/configs/exp-change_antenna_num/antenna-2-power-1-layout-1/rx_config_1.json"
    config_file = "custom_envs/configs/exp-change_antenna_num/antenna-2/rx_config_1.json"
    map_generator = MapGenerator(config_file)
    map_generator.generate_map()
