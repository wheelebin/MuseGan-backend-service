"""This script performs inference from a trained model."""
import os
import time
import logging
import argparse
from pprint import pformat

# from utils.musegan.src import musegan
import numpy as np
import scipy.stats
import tensorflow as tf
from .musegan.config import LOGLEVEL, LOG_FORMAT
from .musegan.data import load_data, get_samples
from .musegan.model import Model
from .musegan.utils import make_sure_path_exists, load_yaml, update_not_none
from .musegan.io_utils import save_pianoroll

from config import ROOT_DIR

LOGGER = logging.getLogger("musegan.inference")


def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", help="Directory where the results are saved.")
    parser.add_argument("--checkpoint_dir", help="Directory that contains checkpoints.")
    parser.add_argument(
        "--params",
        "--params_file",
        "--params_file_path",
        help="Path to the file that defines the " "hyperparameters.",
    )
    parser.add_argument("--config", help="Path to the configuration file.")
    parser.add_argument(
        "--runs", type=int, default="1", help="Times to run the inference process."
    )
    parser.add_argument(
        "--rows", type=int, default=5, help="Number of images per row to be generated."
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=5,
        help="Number of images per column to be generated.",
    )
    parser.add_argument(
        "--lower",
        type=float,
        default=-2,
        help="Lower bound of the truncated normal random " "variables.",
    )
    parser.add_argument(
        "--upper",
        type=float,
        default=2,
        help="Upper bound of the truncated normal random " "variables.",
    )
    parser.add_argument(
        "--gpu",
        "--gpu_device_num",
        type=str,
        default="0",
        help="The GPU device number to use.",
    )
    args = parser.parse_args()
    return args


def get_config_params():
    # This is just lazy
    args = {
        "checkpoint_dir": ROOT_DIR + "/utils/musegan/exp/default/model",
        "columns": 5,
        "config": ROOT_DIR + "/utils/musegan/exp/default/config.yaml",
        "gpu": "1",
        "lower": -2,
        "params": ROOT_DIR + "/utils/musegan/exp/default/params.yaml",
        "result_dir": ROOT_DIR + "/utils/musegan/exp/default/results/inference",
        "rows": 5,
        "runs": 1,
        "upper": 2,
    }

    # Load parameters
    params = load_yaml(args["params"])

    # Load training configurations
    config = load_yaml(args["config"])

    update_not_none(config, args)
    return config, params


def setup():
    """Parse command line arguments, load model parameters, load configurations
    and setup environment."""
    config, params = get_config_params()

    # Set unspecified schedule steps to default values
    # Bellow is not being used
    for target in (config["learning_rate_schedule"], config["slope_schedule"]):
        if target["start"] is None:
            target["start"] = 0
        if target["end"] is None:
            target["end"] = config["steps"]

    # Make sure result directory exists
    make_sure_path_exists(config["result_dir"])

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

    # Setup
    logging.basicConfig(level=LOGLEVEL, format=LOG_FORMAT)

    # params, config = setup()
    LOGGER.info("Using parameters:\n%s", pformat(params))
    LOGGER.info("Using configurations:\n%s", pformat(config))

    # ============================== Placeholders ==============================
    placeholder_x = tf.placeholder(tf.float32, shape=([None] + params["data_shape"]))
    placeholder_z = tf.placeholder(tf.float32, shape=(None, params["latent_dim"]))
    placeholder_c = tf.placeholder(
        tf.float32, shape=([None] + params["data_shape"][:-1] + [1])
    )
    placeholder_suffix = tf.placeholder(tf.string)

    # ================================= Model ==================================
    # Create sampler configurations
    sampler_config = {
        "result_dir": config["result_dir"],
        "image_grid": (config["rows"], config["columns"]),
        "suffix": placeholder_suffix,
        "midi": config["midi"],
        "colormap": np.array(config["colormap"]).T,
        # Save samples
        "collect_save_arrays_op": config["save_array_samples"],
        "collect_save_images_op": config["save_image_samples"],
        "collect_save_pianorolls_op": config["save_pianoroll_samples"],
    }

    # Build model
    model = Model(params)
    if params.get("is_accompaniment"):
        _ = model(
            x=placeholder_x,
            c=placeholder_c,
            z=placeholder_z,
            mode="train",
            params=params,
            config=config,
        )
        predict_nodes = model(
            c=placeholder_c,
            z=placeholder_z,
            mode="predict",
            params=params,
            config=sampler_config,
        )
    else:
        # Training model from models _call_ -> get_train_nodes (Not using the returned nodes)
        _ = model(
            x=placeholder_x, z=placeholder_z, mode="train", params=params, config=config
        )
        # Getting nodes from models _call_ -> get_predict_nodes
        predict_nodes = model(
            z=placeholder_z, mode="predict", params=params, config=sampler_config
        )

    # Get sampler op
    # Sampler op if you want to run several operations (This has a drawback of no results)
    sampler_op = tf.group(
        [
            predict_nodes[key]
            for key in ("save_arrays_op", "save_images_op", "save_pianorolls_op")
            if key in predict_nodes
        ]
    )

    # ================================== Data ==================================
    data = None
    if params.get("is_accompaniment"):
        data = load_data(config["data_source"], config["data_filename"])

    model.get_predict_nodes(
        z=placeholder_z, c=placeholder_c, params=params, config=sampler_config
    )

    return (
        params,
        config,
        predict_nodes["save_pianorolls_op"],  # sampler_op,
        placeholder_c,
        placeholder_z,
        placeholder_suffix,
        data,
    )


class Inference:
    def __init__(self):
        (
            params,
            config,
            sampler_op,
            placeholder_c,
            placeholder_z,
            placeholder_suffix,
            data,
        ) = setup()

        self.config = config
        self.params = params
        self.sampler_op = sampler_op

        # ========================== Session Preparation ===========================
        # Get tensorflow session config
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        # Create saver to restore variables
        saver = tf.train.Saver()

        self.sess = tf.Session(config=tf_config)

        # Restore the latest checkpoint
        LOGGER.info("Restoring the latest checkpoint.")
        with open(os.path.join(config["checkpoint_dir"], "checkpoint")) as f:
            checkpoint_name = os.path.basename(f.readline().split()[1].strip('"'))
        checkpoint_path = os.path.realpath(
            os.path.join(config["checkpoint_dir"], checkpoint_name)
        )
        saver.restore(self.sess, checkpoint_path)

        # Run sampler op
        self.feed_dict_sampler = {
            placeholder_z: scipy.stats.truncnorm.rvs(
                config["lower"],
                config["upper"],
                size=((config["rows"] * config["columns"]), params["latent_dim"]),
            ),
            placeholder_suffix: str(
                0
            ),  # I don't think this is being used, see _save_pianoroll
        }
        if params.get("is_accompaniment"):
            sample_x = get_samples(
                (config["rows"] * config["columns"]),
                data,
                use_random_transpose=config["use_random_transpose"],
            )
            self.feed_dict_sampler[placeholder_c] = np.expand_dims(
                sample_x[..., params["condition_track_idx"]], -1
            )

    def predict(self):
        result = self.sess.run(self.sampler_op, feed_dict=self.feed_dict_sampler)
        array = result > 0
        return array

    def save_pianoroll(self, array, name):
        save_pianoroll(
            name,
            array,
            self.config["midi"]["programs"],
            list(map(bool, self.config["midi"]["is_drums"])),
            self.config["midi"]["tempo"],
            self.params["beat_resolution"],
            self.config["midi"]["lowest_pitch"],
        )


def run_prediction(
    params, config, sampler_op, placeholder_c, placeholder_z, placeholder_suffix, data
):
    start = time.time()
    # ========================== Session Preparation ===========================
    # Get tensorflow session config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Create saver to restore variables
    saver = tf.train.Saver()

    # =========================== Tensorflow Session ===========================
    with tf.Session(config=tf_config) as sess:

        # Restore the latest checkpoint
        LOGGER.info("Restoring the latest checkpoint.")
        with open(os.path.join(config["checkpoint_dir"], "checkpoint")) as f:
            checkpoint_name = os.path.basename(f.readline().split()[1].strip('"'))
        checkpoint_path = os.path.realpath(
            os.path.join(config["checkpoint_dir"], checkpoint_name)
        )
        saver.restore(sess, checkpoint_path)

        # Run sampler op
        for i in range(config["runs"]):
            feed_dict_sampler = {
                placeholder_z: scipy.stats.truncnorm.rvs(
                    config["lower"],
                    config["upper"],
                    size=((config["rows"] * config["columns"]), params["latent_dim"]),
                ),
                placeholder_suffix: str(i),
            }
            if params.get("is_accompaniment"):
                sample_x = get_samples(
                    (config["rows"] * config["columns"]),
                    data,
                    use_random_transpose=config["use_random_transpose"],
                )
                feed_dict_sampler[placeholder_c] = np.expand_dims(
                    sample_x[..., params["condition_track_idx"]], -1
                )

            LOGGER.info("Running operations")
            start_ = time.time()
            result = sess.run(sampler_op, feed_dict=feed_dict_sampler)
            LOGGER.info("Operations run time: %d" % float(time.time() - start_))
            array = result > 0

            LOGGER.info("Saving Pianoroll")
            save_pianoroll(
                "test_01",
                array,
                config["midi"]["programs"],
                list(map(bool, config["midi"]["is_drums"])),
                config["midi"]["tempo"],
                params["beat_resolution"],
                config["midi"]["lowest_pitch"],
            )

            LOGGER.info("Pianoroll saved")

    LOGGER.info("Prediction run time: %d" % float(time.time() - start))


def main():
    """Main function."""
    config, params = get_config_params()

    inference = Inference()

    while True:
        i = input("Run prediction again? [y/n] ")
        if i == "y":
            result = inference.predict()
        else:
            break

    result = inference.predict()

    save_pianoroll(
        "test_03",
        result,
        config["midi"]["programs"],
        list(map(bool, config["midi"]["is_drums"])),
        config["midi"]["tempo"],
        params["beat_resolution"],
        config["midi"]["lowest_pitch"],
    )

    print("Pianoroll saved")

    return
    ##### Bellow is run_prediction method

    (
        params,
        config,
        sampler_op,
        placeholder_c,
        placeholder_z,
        placeholder_suffix,
        data,
    ) = setup()

    while True:
        i = input("Run prediction again? [y/n] ")
        if i == "y":
            run_prediction(
                params,
                config,
                sampler_op,
                placeholder_c,
                placeholder_z,
                placeholder_suffix,
                data,
            )
        else:
            break


if __name__ == "__main__":
    main()
